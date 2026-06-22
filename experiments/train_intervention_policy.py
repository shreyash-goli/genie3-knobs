"""Detect per-target commitment windows and train a PPO policy over conditioning-scale interventions.

Steps:
  1. Estimate per-target commitment windows from offline reward variance by timestep.
  2. Train PPO over direction_scale choices {0.0, 0.5, 1.0, 2.0, 4.0} on DiffusionInterventionEnv
     in offline mode (no GPU needed).
  3. Compare PPO vs fixed scale=1.0 baseline.

Writes: data/experiment_logs/intervention_policy/{summary.json, windows.json}

Run (no GPU needed):
    python -m experiments.train_intervention_policy
"""

from __future__ import annotations

import json
import logging

import numpy as np

import config
from envs.commitment_window import (
    CommitmentWindowDetector,
    DiffusionInterventionEnv,
    INTERVENTION_SCALES,
)
from instrumentation.trajectory_logger import load_records
from oracle.reward_oracle import compute_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _evaluate_fixed_scale(env: DiffusionInterventionEnv, scale: float,
                           n_episodes: int = 200) -> float:
    """Evaluate a fixed direction_scale action."""
    action = INTERVENTION_SCALES.index(scale) if scale in INTERVENTION_SCALES else 2  # default 1.0
    rewards = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=i)
        _, reward, _, _, _ = env.step(action)
        rewards.append(reward)
    return float(np.mean(rewards))


def _train_ppo_intervention(env: DiffusionInterventionEnv,
                             total_timesteps: int = 10_000) -> "PPOPolicyWrapper":
    """Train a PPO policy on DiffusionInterventionEnv (offline mode)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    import torch
    torch.set_num_threads(1)

    import gymnasium as gym
    monitored = Monitor(env)
    model = PPO(
        "MlpPolicy",
        monitored,
        seed=0,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.01,
        n_epochs=10,
        verbose=0,
        policy_kwargs=dict(net_arch=[64, 64]),
        device="cpu",
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model


def _evaluate_ppo(model, env: DiffusionInterventionEnv, n_episodes: int = 200) -> float:
    rewards = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=i + 1000)
        obs_arr = np.array(obs, dtype=np.float32)
        action, _ = model.predict(obs_arr, deterministic=True)
        _, reward, _, _, _ = env.step(int(action))
        rewards.append(reward)
    return float(np.mean(rewards))


def main():
    log_dir = config.EXPERIMENTS_LOG_DIR / "intervention_policy"
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: detect commitment windows ---
    log.info("Detecting per-target commitment windows...")
    records = load_records()
    detector = CommitmentWindowDetector(noise_floor_frac=0.2, min_n=3)
    windows = detector.detect(records)

    windows_json = {
        t: {
            "window_start": w.window_start,
            "window_end": w.window_end,
            "peak_variance_ts": w.peak_variance_ts,
            "variance_by_ts": {str(k): v for k, v in w.variance_by_ts.items()},
        }
        for t, w in windows.items()
    }
    (log_dir / "windows.json").write_text(json.dumps(windows_json, indent=2))
    log.info("Windows saved to %s/windows.json", log_dir)
    for t, w in windows.items():
        log.info("  %s: peak_ts=%d  window=[%d, %d]", t, w.peak_variance_ts,
                 w.window_start, w.window_end)

    # --- Step 2: build DiffusionInterventionEnv (offline) ---
    env = DiffusionInterventionEnv(
        targets=list(windows.keys()) or config.STAGE3_TARGETS,
        commitment_windows=windows,
        oracle_mode="offline",
    )

    # --- Step 3: baseline — fixed scale=1.0 ---
    log.info("Evaluating fixed scale=1.0 baseline...")
    fixed_reward = _evaluate_fixed_scale(env, scale=1.0, n_episodes=200)
    log.info("Fixed scale=1.0: mean_reward=%.4f", fixed_reward)

    # --- Step 4: PPO over intervention actions ---
    log.info("Training PPO over intervention actions (10k steps)...")
    ppo_model = _train_ppo_intervention(env, total_timesteps=10_000)
    ppo_reward = _evaluate_ppo(ppo_model, env, n_episodes=200)
    log.info("PPO: mean_reward=%.4f", ppo_reward)

    summary = {
        "targets": list(windows.keys()),
        "fixed_scale_1_reward": fixed_reward,
        "ppo_reward": ppo_reward,
        "ppo_beats_fixed": ppo_reward > fixed_reward,
        "delta": ppo_reward - fixed_reward,
        "intervention_scales": list(INTERVENTION_SCALES),
        "n_eval_episodes": 200,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Commitment Window + Intervention Policy ===")
    print(f"  fixed scale=1.0  mean_reward = {fixed_reward:.4f}")
    print(f"  PPO              mean_reward = {ppo_reward:.4f}  (Δ={ppo_reward - fixed_reward:+.4f})")
    verdict = "PPO BEATS fixed" if summary["ppo_beats_fixed"] else "PPO does NOT beat fixed"
    print(f"  -> {verdict}")
    print(f"  logs: {log_dir}/")


if __name__ == "__main__":
    main()
