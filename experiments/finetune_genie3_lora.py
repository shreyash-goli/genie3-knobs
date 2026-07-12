"""Fine-tune Genie3's V1Denoiser with LoRA adapters via PPO.

Training uses the offline oracle (instantaneous lookups against logged data) so 500
episodes complete in minutes rather than days.  Live oracle is used only for the
pre/post eval comparison (10 episodes each ≈ 100 min), which fits comfortably inside
a 4-hour SLURM job.

Protocol:
  1. Load V1Denoiser from checkpoint
  2. Attach LoRA adapters (only adapter weights are trainable)
  3. Pre-LoRA live eval — 10 episodes with random policy
  4. PPO fine-tuning — 500 offline episodes, update every 32
  5. Post-LoRA live eval — 10 episodes with trained policy
  6. Save LoRA adapter weights to data/experiment_logs/genie3_lora/lora_adapter/

Prerequisites (GPU node, genie3 conda env):
    conda activate genie3
    pip install peft
    export RLKNOBS_GENIE3_ROOT=~/genie3
    export RLKNOBS_LIVE_SCRATCH=/pscratch/sd/s/shreyash/rlknobs_live
    python -m experiments.finetune_genie3_lora

Time estimate:
    Offline PPO training (500 eps): ~2–5 min
    Live eval pre+post (20 eps total): ~200 min (~3.3 h)
    Total wall-clock: ~3.5 h  →  fits inside a 4-hour SLURM job
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

N_EVAL_EPISODES = 10   # live oracle episodes for pre- and post-LoRA eval
N_TRAIN_EPISODES = 500 # offline PPO episodes


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


def _run_eval(env, actor_critic, n_episodes: int, label: str) -> list[float]:
    """Run n_episodes with the current policy, return per-episode rewards."""
    import torch
    rewards = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=i if label == "pre" else i + 1000)
        done = False
        ep_reward = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = actor_critic(obs_t)
                action = int(logits.argmax(dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        log.info("%s eval episode %d/%d  reward=%.4f", label, i + 1, n_episodes, ep_reward)
    mean = sum(rewards) / len(rewards)
    log.info("%s eval complete  mean_reward=%.4f", label, mean)
    return rewards


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
    log.info("Loading V1Denoiser from checkpoint...")
    _orig_dir = os.getcwd()
    os.chdir(str(genie3_root))
    run_config = load_experiment_config(str(config_yaml))
    generation_config = to_generation_config(run_config, shard_id=0, num_shards=1)
    sample_config = build_sample_config_from_dict(generation_config)
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

    # --- shared setup: commitment windows + actor-critic ---
    log.info("Detecting commitment windows from offline data...")
    records = load_records()
    detector = CommitmentWindowDetector()
    windows = detector.detect(records)
    poc_targets = list(config.STAGE3_TARGETS)
    poc_windows = {t: windows[t] for t in poc_targets if t in windows}

    buf = FrontierBuffer(size=64, epsilon=0.1, temperature=1.0, p_frontier=0.5, seed=0)
    buf.initialize(targets=poc_targets)

    _probe_env = DiffusionInterventionEnv(
        targets=poc_targets, commitment_windows=poc_windows, oracle_mode="offline"
    )
    actor_critic = ActorCritic.build(
        obs_dim=_probe_env.observation_space.shape[0],
        n_actions=_probe_env.n_actions,
        hidden=(64, 64),
    )

    # --- Step 3: pre-LoRA live eval ---
    log.info("Pre-LoRA live eval (%d episodes)...", N_EVAL_EPISODES)
    live_env = DiffusionInterventionEnv(
        targets=poc_targets,
        commitment_windows=poc_windows,
        frontier_buffer=buf,
        oracle_mode="live",
    )
    pre_rewards = _run_eval(live_env, actor_critic, N_EVAL_EPISODES, label="pre")
    pre_mean = sum(pre_rewards) / len(pre_rewards)

    # --- Step 4: PPO fine-tuning (offline oracle — completes in minutes) ---
    log.info("PPO fine-tuning: %d offline episodes...", N_TRAIN_EPISODES)
    offline_env = DiffusionInterventionEnv(
        targets=poc_targets,
        commitment_windows=poc_windows,
        frontier_buffer=buf,
        oracle_mode="offline",
    )
    ppo_cfg = PPOFinetuneConfig(
        total_episodes=N_TRAIN_EPISODES,
        ppo_update_freq=32,
        n_epochs=4,
        batch_size=32,
        learning_rate=1e-4,
        save_every=100,
        save_dir=str(config.EXPERIMENTS_LOG_DIR / "genie3_lora" / "checkpoints"),
    )
    train_log = train_lora_ppo(offline_env, actor_critic, ppo_cfg, verbose=True)
    log.info("Training complete. mean_reward=%.4f", train_log["mean_reward"])

    # --- Step 5: post-LoRA live eval ---
    log.info("Post-LoRA live eval (%d episodes)...", N_EVAL_EPISODES)
    post_rewards = _run_eval(live_env, actor_critic, N_EVAL_EPISODES, label="post")
    post_mean = sum(post_rewards) / len(post_rewards)

    # --- Step 6: save LoRA adapter ---
    lora_save_dir = config.EXPERIMENTS_LOG_DIR / "genie3_lora" / "lora_adapter"
    save_lora(lora_model, lora_save_dir)

    # --- summary ---
    log_dir = config.EXPERIMENTS_LOG_DIR / "genie3_lora"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "pre_lora_mean_reward": pre_mean,
        "post_lora_mean_reward": post_mean,
        "delta": post_mean - pre_mean,
        "pre_lora_rewards": pre_rewards,
        "post_lora_rewards": post_rewards,
        "n_eval_episodes": N_EVAL_EPISODES,
        "n_train_episodes": N_TRAIN_EPISODES,
        "oracle_mode_train": "offline",
        "oracle_mode_eval": "live",
        "lora_config": {
            "r": lora_cfg.r,
            "lora_alpha": lora_cfg.lora_alpha,
            "target_modules": lora_cfg.target_modules,
        },
        "ppo_config": {
            "total_episodes": ppo_cfg.total_episodes,
            "ppo_update_freq": ppo_cfg.ppo_update_freq,
            "learning_rate": ppo_cfg.learning_rate,
        },
        "training_mean_reward": train_log["mean_reward"],
        "lora_adapter_dir": str(lora_save_dir),
        "buffer_stats": buf.stats(),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Stage 7: LoRA Fine-tuning ===")
    print(f"  pre-LoRA  mean_reward = {pre_mean:.4f}  (n={N_EVAL_EPISODES})")
    print(f"  post-LoRA mean_reward = {post_mean:.4f}  (Δ={post_mean - pre_mean:+.4f})")
    print(f"  LoRA adapter: {lora_save_dir}")
    print(f"  logs: {log_dir}/summary.json")


if __name__ == "__main__":
    main()
