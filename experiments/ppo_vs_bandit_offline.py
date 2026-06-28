"""PPO vs bandit comparison — fully offline, no GPU required.

Trains a PPO policy on the windowed 10-step MDP using the offline oracle (real logged
ColabFold data), then compares it against two bandit baselines on held-out episodes.

What this validates:
    Does learning *which* hotspot mode to apply at *which* diffusion timestep within the
    commitment window actually improve outcomes over a fixed-choice baseline?

Baselines:
    random  — picks a hotspot mode uniformly at random at each step (no learning)
    fixed-* — always picks the same hotspot mode for all 10 steps; one run per mode
              (all / ablate_competitors / missed_only); best of the three is the
              "optimal bandit"

Usage:
    cd /global/u2/s/shreyash/ppo-idea
    conda run -n genie3 python -m experiments.ppo_vs_bandit_offline

    # or with custom settings:
    N_TRAIN=500 N_EVAL=200 conda run -n genie3 python -m experiments.ppo_vs_bandit_offline

Output:
    data/experiment_logs/ppo_vs_bandit/results.json
    data/experiment_logs/ppo_vs_bandit/training_curve.json

Time estimate:
    ~2–10 min total (all offline, no ColabFold calls)
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from pathlib import Path
from typing import Any

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

N_TRAIN = int(os.environ.get("N_TRAIN", 500))
N_EVAL  = int(os.environ.get("N_EVAL", 200))


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_policy(env, policy_fn, n_episodes: int, seed_offset: int = 0) -> list[float]:
    """Roll out policy_fn for n_episodes; return per-episode terminal rewards."""
    rewards = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + i)
        done = False
        ep_reward = 0.0
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
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


def _fixed_policy(action: int):
    """Always picks the same action regardless of state."""
    def fn(obs):
        return action
    return fn


def _random_policy(rng: np.random.Generator):
    def fn(obs):
        return int(rng.integers(len(HOTSPOT_MODES)))
    return fn


def _summarise(rewards: list[float], label: str) -> dict[str, Any]:
    mean = statistics.mean(rewards)
    std  = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    log.info("%-30s  mean=%.4f  std=%.4f  n=%d", label, mean, std, len(rewards))
    return {"label": label, "mean": mean, "std": std, "n": len(rewards), "rewards": rewards}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("PPO vs bandit — offline eval  (N_TRAIN=%d  N_EVAL=%d)", N_TRAIN, N_EVAL)

    # --- shared env setup ---
    records = load_records()
    detector = CommitmentWindowDetector()
    windows  = detector.detect(records)
    targets  = list(config.STAGE3_TARGETS)
    cw       = {t: windows[t] for t in targets if t in windows}

    intermediate_reward_scale = float(os.environ.get("INTERMEDIATE_REWARD_SCALE", 0.1))
    log.info("intermediate_reward_scale=%.2f", intermediate_reward_scale)

    def make_env():
        return DiffusionInterventionEnv(
            targets=targets, commitment_windows=cw, oracle_mode="offline",
            intermediate_reward_scale=intermediate_reward_scale, seed=42,
        )

    train_env = make_env()
    eval_env  = make_env()

    obs_dim  = train_env.observation_space.shape[0]
    n_actions = train_env.n_actions

    # --- PPO training ---
    log.info("Training PPO: %d episodes, %d steps/episode ...", N_TRAIN, N_WINDOW_STEPS)
    actor_critic = ActorCritic.build(obs_dim=obs_dim, n_actions=n_actions, hidden=(64, 64))
    ppo_cfg = PPOFinetuneConfig(
        total_episodes=N_TRAIN,
        ppo_update_freq=32,
        n_epochs=4,
        batch_size=32,
        learning_rate=3e-4,
        save_every=N_TRAIN + 1,  # no mid-run checkpoints needed
        save_dir=str(config.EXPERIMENTS_LOG_DIR / "ppo_vs_bandit" / "checkpoints"),
    )
    train_log = train_lora_ppo(train_env, actor_critic, ppo_cfg, verbose=True)
    log.info("Training done. mean_reward=%.4f", train_log["mean_reward"])

    # --- Eval: PPO ---
    log.info("Evaluating policies on %d held-out episodes ...", N_EVAL)
    # Use seed_offset=N_TRAIN so eval episodes are disjoint from training seeds
    ppo_rewards = _eval_policy(eval_env, _ppo_policy(actor_critic), N_EVAL, seed_offset=N_TRAIN)

    # --- Eval: random bandit ---
    rng = np.random.default_rng(0)
    random_rewards = _eval_policy(eval_env, _random_policy(rng), N_EVAL, seed_offset=N_TRAIN)

    # --- Eval: fixed bandits (one per hotspot mode) ---
    fixed_results = {}
    for i, mode in enumerate(HOTSPOT_MODES):
        r = _eval_policy(eval_env, _fixed_policy(i), N_EVAL, seed_offset=N_TRAIN)
        fixed_results[mode] = r

    best_fixed_mode = max(fixed_results, key=lambda m: statistics.mean(fixed_results[m]))
    best_fixed_rewards = fixed_results[best_fixed_mode]

    # --- Summary ---
    results = {
        "ppo":          _summarise(ppo_rewards,        "PPO (trained)"),
        "random":       _summarise(random_rewards,     "random bandit"),
        "best_fixed":   _summarise(best_fixed_rewards, f"fixed bandit ({best_fixed_mode})"),
        "fixed_by_mode": {
            mode: _summarise(fixed_results[mode], f"fixed ({mode})")
            for mode in HOTSPOT_MODES
        },
        "training": {
            "n_episodes":    N_TRAIN,
            "n_eval":        N_EVAL,
            "n_window_steps": N_WINDOW_STEPS,
            "oracle_mode":   "offline",
            "training_mean_reward": train_log["mean_reward"],
            "training_curve": train_log.get("episode_rewards", []),
        },
        "delta_ppo_vs_best_fixed": (
            statistics.mean(ppo_rewards) - statistics.mean(best_fixed_rewards)
        ),
        "delta_ppo_vs_random": (
            statistics.mean(ppo_rewards) - statistics.mean(random_rewards)
        ),
    }

    out_dir = config.EXPERIMENTS_LOG_DIR / "ppo_vs_bandit"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    log.info("Results written to %s/results.json", out_dir)

    # --- Print table ---
    print(f"\n{'='*55}")
    print(f"  PPO vs bandit — windowed MDP ({N_WINDOW_STEPS} steps/episode)")
    print(f"  Offline oracle  |  N_eval={N_EVAL} episodes")
    print(f"{'='*55}")
    rows = [
        ("PPO (trained)",                 ppo_rewards),
        (f"best fixed ({best_fixed_mode})", best_fixed_rewards),
        ("random bandit",                 random_rewards),
    ]
    for label, r in rows:
        mean = statistics.mean(r)
        std  = statistics.stdev(r) if len(r) > 1 else 0.0
        print(f"  {label:<30}  {mean:.4f} ± {std:.4f}")
    print(f"{'='*55}")
    ppo_mean = statistics.mean(ppo_rewards)
    bf_mean  = statistics.mean(best_fixed_rewards)
    print(f"  PPO vs best fixed:  {ppo_mean - bf_mean:+.4f}")
    print(f"  PPO vs random:      {ppo_mean - statistics.mean(random_rewards):+.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
