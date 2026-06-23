"""Fine-tune Genie3's V1Denoiser with LoRA adapters via PPO (live oracle, GPU required).

Attaches PEFT LoRA adapters to IPA attention layers (linear_q, linear_kv, linear_out)
and LatentTransformer pair projections (linear_pi, linear_pj, linear_p), then runs PPO
over the conditioning-scale intervention action space with FrontierBuffer seeding.

Protocol:
  1. Load V1Denoiser from checkpoint
  2. Attach LoRA adapters (only adapter weights are trainable)
  3. Build DiffusionInterventionEnv in live mode with FrontierBuffer seeding
  4. PPO fine-tuning: 500 episodes, update every 32
  5. Compare pre-LoRA vs post-LoRA reward on 10 held-out episodes
  6. Save LoRA adapter weights to data/experiment_logs/genie3_lora/lora_adapter/

Prerequisites (GPU node, genie3 conda env):
    conda activate genie3
    pip install peft
    export RLKNOBS_GENIE3_ROOT=~/genie3
    export RLKNOBS_LIVE_SCRATCH=/pscratch/sd/s/shreyash/rlknobs_live
    python -m experiments.finetune_genie3_lora
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _check_prerequisites():
    missing = []
    try:
        import genie3  # noqa
    except ImportError:
        missing.append("genie3 (run inside genie3 conda env)")
    try:
        import peft  # noqa
    except ImportError:
        missing.append("peft (pip install peft)")
    if missing:
        log.error("Missing prerequisites:\n  " + "\n  ".join(missing))
        sys.exit(1)


def main():
    _check_prerequisites()

    import torch
    from genie3.config import load_experiment_config, to_generation_config
    from genie3.generation.config.registry import build_sample_config_from_dict
    from genie3.generation.model.registry import get_model
    from collections import OrderedDict

    from buffer.frontier_buffer import FrontierBuffer
    from envs.commitment_window import CommitmentWindowDetector, DiffusionInterventionEnv
    from instrumentation.trajectory_logger import load_records
    from policy.lora_finetune import (
        LoRAConfig, PPOFinetuneConfig, attach_lora, save_lora, train_lora_ppo, ActorCritic,
    )

    import os
    genie3_root = Path(os.environ.get("RLKNOBS_GENIE3_ROOT", Path.home() / "genie3"))
    config_yaml = genie3_root / "branching" / "configs" / "experiment_trajectory_branching.yaml"

    # --- Step 1: load V1Denoiser ---
    # genie3 config YAMLs use relative paths (pretrained/v1/config.yaml etc.)
    # that are relative to the genie3 repo root — cd there before loading.
    log.info("Loading V1Denoiser from checkpoint...")
    _orig_dir = os.getcwd()
    os.chdir(str(genie3_root))
    run_config = load_experiment_config(str(config_yaml))
    generation_config = to_generation_config(run_config, shard_id=0, num_shards=1)
    sample_config = build_sample_config_from_dict(generation_config)
    # resolve all relative paths to absolute while still inside genie3_root
    checkpoint_path = str((genie3_root / sample_config.base.checkpoint).resolve())
    os.chdir(_orig_dir)

    model = get_model(sample_config.model.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    updated_sd = OrderedDict()
    for k, v in state_dict.items():
        k2 = k.replace("_orig_mod.", "").replace(".linear_motif_template.", ".linear_cond_template.")
        if k2.startswith("model."):
            k2 = k2[len("model."):]
        updated_sd[k2] = v
    model.load_state_dict(updated_sd, strict=True)
    log.info("Checkpoint loaded: %s", checkpoint_path)

    # --- Step 2: attach LoRA adapters ---
    lora_cfg = LoRAConfig.from_env()
    log.info("Attaching LoRA adapters: r=%d  alpha=%d  targets=%s",
             lora_cfg.r, lora_cfg.lora_alpha, lora_cfg.target_modules)
    lora_model = attach_lora(model, lora_cfg)

    # --- Step 3: build env with live oracle + frontier buffer ---
    log.info("Detecting commitment windows from offline data...")
    records = load_records()
    detector = CommitmentWindowDetector()
    windows = detector.detect(records)

    # restrict to contrast targets for the proof-of-concept run (26 oracle calls ~2h)
    poc_targets = list(config.STAGE3_TARGETS)
    poc_windows = {t: windows[t] for t in poc_targets if t in windows}

    buf = FrontierBuffer(size=64, epsilon=0.1, temperature=1.0, p_frontier=0.5, seed=0)
    buf.initialize(targets=poc_targets)

    env = DiffusionInterventionEnv(
        config_yaml=str(config_yaml),
        targets=poc_targets,
        commitment_windows=poc_windows,
        frontier_buffer=buf,
        oracle_mode="live",
    )

    # --- Step 4: pre-LoRA baseline (3 episodes for quick proof-of-concept) ---
    log.info("Evaluating pre-LoRA baseline (3 episodes)...")
    pre_rewards = []
    for i in range(3):
        obs, info = env.reset(seed=i)
        action = env.action_space.sample()  # random baseline
        _, reward, _, _, _ = env.step(action)
        pre_rewards.append(reward)
    pre_mean = sum(pre_rewards) / len(pre_rewards)
    log.info("Pre-LoRA mean_reward=%.4f", pre_mean)

    # --- Step 5: PPO fine-tuning ---
    obs_dim = env.observation_space.shape[0]
    actor_critic = ActorCritic.build(obs_dim=obs_dim, n_actions=env.n_actions, hidden=(64, 64))

    ppo_cfg = PPOFinetuneConfig(
        total_episodes=20,      # proof-of-concept; increase to 500 for full training
        ppo_update_freq=10,     # update every 10 episodes at this scale
        n_epochs=4,
        batch_size=8,
        learning_rate=1e-4,
        save_every=10,
        save_dir=str(config.EXPERIMENTS_LOG_DIR / "genie3_lora" / "checkpoints"),
    )

    log.info("Starting PPO fine-tuning (%d episodes)...", ppo_cfg.total_episodes)
    train_log = train_lora_ppo(env, actor_critic, ppo_cfg, verbose=True)
    log.info("Training complete. mean_reward=%.4f", train_log["mean_reward"])

    # --- Step 6: post-LoRA evaluation (3 episodes, matching pre-LoRA baseline) ---
    log.info("Evaluating post-LoRA policy (3 episodes)...")
    post_rewards = []
    import numpy as np
    import torch as _torch
    for i in range(3):
        obs, _ = env.reset(seed=i + 500)
        obs_t = _torch.tensor(obs, dtype=_torch.float32).unsqueeze(0)
        with _torch.no_grad():
            logits, _ = actor_critic(obs_t)
            action = int(logits.argmax(dim=-1).item())
        _, reward, _, _, _ = env.step(action)
        post_rewards.append(reward)
    post_mean = sum(post_rewards) / len(post_rewards)
    log.info("Post-LoRA mean_reward=%.4f", post_mean)

    # --- Step 7: save LoRA adapter ---
    lora_save_dir = config.EXPERIMENTS_LOG_DIR / "genie3_lora" / "lora_adapter"
    save_lora(lora_model, lora_save_dir)

    # --- summary ---
    log_dir = config.EXPERIMENTS_LOG_DIR / "genie3_lora"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "pre_lora_mean_reward": pre_mean,
        "post_lora_mean_reward": post_mean,
        "delta": post_mean - pre_mean,
        "lora_config": {
            "r": lora_cfg.r,
            "lora_alpha": lora_cfg.lora_alpha,
            "target_modules": lora_cfg.target_modules,
        },
        "ppo_config": {
            "total_episodes": ppo_cfg.total_episodes,
            "learning_rate": ppo_cfg.learning_rate,
        },
        "training_mean_reward": train_log["mean_reward"],
        "lora_adapter_dir": str(lora_save_dir),
        "buffer_stats": buf.stats(),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Stage 7: LoRA Fine-tuning ===")
    print(f"  pre-LoRA  mean_reward = {pre_mean:.4f}")
    print(f"  post-LoRA mean_reward = {post_mean:.4f}  (Δ={post_mean - pre_mean:+.4f})")
    print(f"  LoRA adapter: {lora_save_dir}")
    print(f"  logs: {log_dir}/summary.json")


if __name__ == "__main__":
    main()
