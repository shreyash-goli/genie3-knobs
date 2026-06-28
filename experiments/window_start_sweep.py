"""Window-start placement sweep — which window_start gives the best PPO outcome?

Motivated by the guidance-interval paper (Kynkäänniemi et al., NeurIPS 2024):
    - window_start (when to begin intervening, higher t = earlier in diffusion)
      is the unforgiving boundary: too early oversteers before fold class commits,
      too late leaves insufficient propagation time.
    - window_end (when to stop) is forgiving — sweep it last, expect flat curve.

This experiment fixes window_end=700 and sweeps window_start ∈ WINDOW_START_CANDIDATES,
training a fresh PPO policy for each placement and recording mean eval reward.  PPO uses
intermediate_reward_scale=0.1 (iCS proxy) so credit assignment is tractable.

Usage:
    cd /global/u2/s/shreyash/ppo-idea
    conda run -n genie3 python -m experiments.window_start_sweep

    # custom budget:
    N_TRAIN=1000 N_EVAL=200 conda run -n genie3 python -m experiments.window_start_sweep

Output:
    data/experiment_logs/window_start_sweep/results.json

Time estimate:
    ~10–30 min total (all offline)
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from pathlib import Path

import numpy as np

import config
from envs.commitment_window import (
    CommitmentWindowDetector,
    DiffusionInterventionEnv,
    HOTSPOT_MODES,
    N_WINDOW_STEPS,
)
from instrumentation.trajectory_logger import load_records
from policy.lora_finetune import ActorCritic, PPOFinetuneConfig, train_lora_ppo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

N_TRAIN = int(os.environ.get("N_TRAIN", 1000))
N_EVAL  = int(os.environ.get("N_EVAL",  200))
INTERMEDIATE_REWARD_SCALE = float(os.environ.get("INTERMEDIATE_REWARD_SCALE", 0.1))

# window_end fixed; sweep window_start (higher t = earlier in diffusion)
WINDOW_END_FIXED    = 700
WINDOW_START_CANDIDATES = [750, 800, 850, 900, 950]


def _eval_policy(env, policy_fn, n_episodes: int, seed_offset: int = 0) -> list[float]:
    rewards = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + i)
        done = False
        ep_reward = 0.0
        while not done:
            action = policy_fn(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_reward += r
            done = terminated or truncated
        rewards.append(ep_reward)
    return rewards


def _ppo_policy(actor_critic):
    import torch
    def fn(obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = actor_critic(obs_t)
        return int(logits.argmax(dim=-1).item())
    return fn


def _fixed_best_policy(env, n_probe: int = 50, seed_offset: int = 9999) -> tuple[int, float]:
    """Find the best fixed action by brute-force eval; used as the per-placement baseline."""
    best_action, best_mean = 0, -float("inf")
    for a in range(len(HOTSPOT_MODES)):
        rewards = []
        for i in range(n_probe):
            obs, _ = env.reset(seed=seed_offset + i)
            done = False
            ep_r = 0.0
            while not done:
                obs, r, terminated, truncated, _ = env.step(a)
                ep_r += r
                done = terminated or truncated
            rewards.append(ep_r)
        mean = statistics.mean(rewards)
        if mean > best_mean:
            best_mean, best_action = mean, a
    return best_action, best_mean


def run_placement(
    window_start: int,
    targets: list[str],
    commitment_windows: dict,
    obs_dim: int,
    n_actions: int,
) -> dict:
    log.info("--- window_start=%d  window_end=%d ---", window_start, WINDOW_END_FIXED)

    def make_env(seed=42):
        return DiffusionInterventionEnv(
            targets=targets,
            commitment_windows=commitment_windows,
            oracle_mode="offline",
            intermediate_reward_scale=INTERMEDIATE_REWARD_SCALE,
            window_start_override=window_start,
            window_end_override=WINDOW_END_FIXED,
            seed=seed,
        )

    train_env = make_env(seed=42)
    eval_env  = make_env(seed=0)

    actor_critic = ActorCritic.build(obs_dim=obs_dim, n_actions=n_actions, hidden=(64, 64))
    ppo_cfg = PPOFinetuneConfig(
        total_episodes=N_TRAIN,
        ppo_update_freq=32,
        n_epochs=4,
        batch_size=32,
        learning_rate=3e-4,
        save_every=N_TRAIN + 1,
    )
    train_log = train_lora_ppo(train_env, actor_critic, ppo_cfg, verbose=False)

    ppo_rewards = _eval_policy(eval_env, _ppo_policy(actor_critic), N_EVAL, seed_offset=N_TRAIN)
    ppo_mean = statistics.mean(ppo_rewards)
    ppo_std  = statistics.stdev(ppo_rewards) if len(ppo_rewards) > 1 else 0.0

    _, fixed_mean = _fixed_best_policy(make_env(seed=1))

    log.info(
        "window_start=%d  ppo=%.4f±%.4f  fixed_best=%.4f  delta=%+.4f",
        window_start, ppo_mean, ppo_std, fixed_mean, ppo_mean - fixed_mean,
    )

    return {
        "window_start": window_start,
        "window_end": WINDOW_END_FIXED,
        "ppo_mean": ppo_mean,
        "ppo_std": ppo_std,
        "fixed_best_mean": fixed_mean,
        "delta_ppo_vs_fixed": ppo_mean - fixed_mean,
        "training_curve": train_log.get("episode_rewards", []),
        "training_mean_reward": train_log["mean_reward"],
    }


def main():
    log.info(
        "Window-start sweep  N_TRAIN=%d  N_EVAL=%d  scale=%.2f",
        N_TRAIN, N_EVAL, INTERMEDIATE_REWARD_SCALE,
    )

    records = load_records()
    detector = CommitmentWindowDetector()
    windows  = detector.detect(records)
    targets  = list(config.STAGE3_TARGETS)
    cw       = {t: windows[t] for t in targets if t in windows}

    # obs_dim from a throwaway env (window bounds don't affect obs shape)
    probe_env = DiffusionInterventionEnv(targets=targets, oracle_mode="offline")
    obs_dim   = probe_env.observation_space.shape[0]
    n_actions = probe_env.n_actions

    sweep_results = []
    for ws in WINDOW_START_CANDIDATES:
        result = run_placement(ws, targets, cw, obs_dim, n_actions)
        sweep_results.append(result)

    out_dir = config.EXPERIMENTS_LOG_DIR / "window_start_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "n_train": N_TRAIN,
            "n_eval": N_EVAL,
            "intermediate_reward_scale": INTERMEDIATE_REWARD_SCALE,
            "window_end_fixed": WINDOW_END_FIXED,
            "window_start_candidates": WINDOW_START_CANDIDATES,
        },
        "results": sweep_results,
    }
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    log.info("Results written to %s/results.json", out_dir)

    print(f"\n{'='*60}")
    print(f"  Window-start sweep  (window_end fixed={WINDOW_END_FIXED})")
    print(f"  N_train={N_TRAIN}  N_eval={N_EVAL}  intermediate_scale={INTERMEDIATE_REWARD_SCALE}")
    print(f"{'='*60}")
    print(f"  {'window_start':>12}  {'PPO mean':>10}  {'fixed best':>10}  {'delta':>8}")
    print(f"  {'-'*46}")
    best = max(sweep_results, key=lambda r: r["ppo_mean"])
    for r in sweep_results:
        marker = " <-- best" if r["window_start"] == best["window_start"] else ""
        print(f"  {r['window_start']:>12}  {r['ppo_mean']:>10.4f}  {r['fixed_best_mean']:>10.4f}  {r['delta_ppo_vs_fixed']:>+8.4f}{marker}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
