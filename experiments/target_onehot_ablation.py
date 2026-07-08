"""Target-one-hot ablation (NEXT_STEPS.md §3.1 follow-up).

The windowed-MDP PPO win in ppo_vs_bandit_offline only appears in the *joint* 2-target
setting; every single-target run (window_start_ablation.py) lost to a constant action.
The leading hypothesis: PPO's advantage is cross-target specialization -- the observation
includes a target one-hot, so a state-conditioned policy can pick different behavior per
target, which a single fixed action cannot. This is the same mechanism behind the Stage 1-3
wins, not evidence for within-episode sequential learning.

This experiment tests that directly: train the joint 2-target env twice, identically, once
with the target one-hot intact (baseline) and once with it masked to all-zeros
(mask_target_onehot=True). If PPO's margin over fixed/random collapses to ~0 when the
one-hot is removed, cross-target specialization is confirmed as the entire effect.

Same setup as ppo_vs_bandit_offline (helpers imported from it): offline oracle, N_TRAIN
train episodes, N_EVAL held-out episodes, intermediate_reward_scale from env.

Usage:
    conda run -n genie3 python -m experiments.target_onehot_ablation
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from typing import Any

import numpy as np

import config
from envs.commitment_window import (
    CommitmentWindowDetector,
    DiffusionInterventionEnv,
    HOTSPOT_MODES,
    N_WINDOW_STEPS,
)
from experiments.ppo_vs_bandit_offline import (
    N_EVAL,
    N_TRAIN,
    _eval_policy,
    _fixed_policy,
    _ppo_policy,
    _random_policy,
    _summarise,
)
from instrumentation.trajectory_logger import load_records
from policy.lora_finetune import ActorCritic, PPOFinetuneConfig, train_lora_ppo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_condition(label: str, mask_target_onehot: bool, targets, cw,
                  intermediate_reward_scale: float) -> dict[str, Any]:
    log.info("=== condition: %s (mask_target_onehot=%s) ===", label, mask_target_onehot)

    def make_env():
        return DiffusionInterventionEnv(
            targets=targets, commitment_windows=cw, oracle_mode="offline",
            intermediate_reward_scale=intermediate_reward_scale,
            mask_target_onehot=mask_target_onehot, seed=42,
        )

    train_env = make_env()
    eval_env = make_env()
    obs_dim = train_env.observation_space.shape[0]
    n_actions = train_env.n_actions

    actor_critic = ActorCritic.build(obs_dim=obs_dim, n_actions=n_actions, hidden=(64, 64))
    ppo_cfg = PPOFinetuneConfig(
        total_episodes=N_TRAIN, ppo_update_freq=32, n_epochs=4, batch_size=32,
        learning_rate=3e-4, save_every=N_TRAIN + 1,
        save_dir=str(config.EXPERIMENTS_LOG_DIR / "target_onehot_ablation" / "checkpoints"),
    )
    train_lora_ppo(train_env, actor_critic, ppo_cfg, verbose=False)

    ppo_rewards = _eval_policy(eval_env, _ppo_policy(actor_critic), N_EVAL, seed_offset=N_TRAIN)
    rng = np.random.default_rng(0)
    random_rewards = _eval_policy(eval_env, _random_policy(rng), N_EVAL, seed_offset=N_TRAIN)
    fixed_results = {
        mode: _eval_policy(eval_env, _fixed_policy(i), N_EVAL, seed_offset=N_TRAIN)
        for i, mode in enumerate(HOTSPOT_MODES)
    }
    best_fixed_mode = max(fixed_results, key=lambda m: statistics.mean(fixed_results[m]))
    best_fixed_rewards = fixed_results[best_fixed_mode]

    ppo_mean = statistics.mean(ppo_rewards)
    fixed_mean = statistics.mean(best_fixed_rewards)
    random_mean = statistics.mean(random_rewards)

    result = {
        "label": label,
        "mask_target_onehot": mask_target_onehot,
        "ppo": _summarise(ppo_rewards, f"{label}: PPO"),
        "random": _summarise(random_rewards, f"{label}: random"),
        "best_fixed": _summarise(best_fixed_rewards, f"{label}: fixed ({best_fixed_mode})"),
        "best_fixed_mode": best_fixed_mode,
        "delta_ppo_vs_fixed": ppo_mean - fixed_mean,
        "delta_ppo_vs_random": ppo_mean - random_mean,
    }
    log.info(
        "%s: PPO=%.4f  fixed=%.4f  random=%.4f  |  delta_vs_fixed=%+.4f  delta_vs_random=%+.4f",
        label, ppo_mean, fixed_mean, random_mean,
        result["delta_ppo_vs_fixed"], result["delta_ppo_vs_random"],
    )
    return result


def main():
    records = load_records()
    windows = CommitmentWindowDetector().detect(records)
    targets = list(config.STAGE3_TARGETS)
    cw = {t: windows[t] for t in targets if t in windows}
    intermediate_reward_scale = float(os.environ.get("INTERMEDIATE_REWARD_SCALE", 0.1))

    log.info(
        "Target-one-hot ablation  N_TRAIN=%d  N_EVAL=%d  targets=%s  scale=%.2f",
        N_TRAIN, N_EVAL, targets, intermediate_reward_scale,
    )

    baseline = run_condition("baseline (one-hot intact)", False, targets, cw,
                             intermediate_reward_scale)
    masked = run_condition("masked (one-hot removed)", True, targets, cw,
                          intermediate_reward_scale)

    out_dir = config.EXPERIMENTS_LOG_DIR / "target_onehot_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {"n_train": N_TRAIN, "n_eval": N_EVAL, "targets": targets,
                   "intermediate_reward_scale": intermediate_reward_scale},
        "baseline": baseline,
        "masked": masked,
    }
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'=' * 68}")
    print("  Target-one-hot ablation -- does PPO's win survive without target identity?")
    print(f"{'=' * 68}")
    print(f"  {'condition':<26}  {'PPO':>8}  {'fixed':>8}  {'random':>8}  {'Δ vs fix':>9}")
    print(f"  {'-' * 62}")
    for c in (baseline, masked):
        print(f"  {c['label']:<26}  {c['ppo']['mean']:>8.4f}  "
              f"{c['best_fixed']['mean']:>8.4f}  {c['random']['mean']:>8.4f}  "
              f"{c['delta_ppo_vs_fixed']:>+9.4f}")
    print(f"{'=' * 68}")
    collapse = masked["delta_ppo_vs_fixed"]
    baseline_margin = baseline["delta_ppo_vs_fixed"]
    print(f"  baseline PPO margin over fixed:  {baseline_margin:+.4f}")
    print(f"  masked   PPO margin over fixed:  {collapse:+.4f}")
    print(f"  => {'COLLAPSED (cross-target specialization confirmed)' if collapse <= baseline_margin * 0.5 else 'margin survived (residual within-episode signal?)'}")
    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    main()
