"""FK Steering vs PPO -- the §7.3 four-way (2×2) comparison, fully offline, no GPU.

Trains a PPO policy on the windowed MDP (offline oracle), then evaluates the §7.3 table:

                     |  No FK (k=1)      |  + FK (k particles)
    -----------------+-------------------+---------------------------
    Random cond.     |  (1) floor        |  (3) search-only
    PPO policy       |  (2) Stage 1-3    |  (4) policy + search

Interpretation (§7.3):
    4 > 2 > 3 > 1  -> the policy adds value *beyond* inference-time search; complementary.
    3 ~= 2         -> learning and search solve the same problem; the question becomes
                      compute-efficiency (policy at 1x vs FK at kx per-sample cost).
Both are reportable -- running the comparison is the deliverable, not which way it lands.

FK here is the offline best-of-k first cut (score particles at t=0 with the terminal reward);
see policy/fk_steering.py for scope and the deferred mid-trajectory direction.

Usage:
    cd /global/u2/s/shreyash/ppo-idea
    conda run -n genie3 python -m experiments.fk_vs_ppo_offline
    N_TRAIN=500 N_EVAL=200 FK_K=8 conda run -n genie3 python -m experiments.fk_vs_ppo_offline
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from typing import Any, Optional

import numpy as np

import config
from envs.commitment_window import (
    CommitmentWindowDetector,
    DiffusionInterventionEnv,
    N_WINDOW_STEPS,
)
from instrumentation.trajectory_logger import load_records
from policy.fk_steering import fk_rollout
from policy.lora_finetune import ActorCritic, PPOFinetuneConfig, train_lora_ppo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

N_TRAIN = int(os.environ.get("N_TRAIN", 500))
N_EVAL  = int(os.environ.get("N_EVAL", 200))
# k values to sweep. 1 is the no-FK baseline (built-in sanity check: FK with k=1 is a single
# trajectory). The largest k is used for the headline 2×2 "+FK" column.
FK_KS = tuple(int(x) for x in os.environ.get("FK_KS", "1,4,8").split(","))
FK_TEMPERATURE = float(os.environ.get("FK_TEMPERATURE", 0.0))  # 0.0 = best-of-k
INDEPENDENT_ACTIONS = os.environ.get("FK_INDEPENDENT", "0") == "1"


def _eval_fk(make_env_seeded, actor_critic, k, n_episodes, seed_offset, temperature):
    """Mean/std of best-survivor terminal reward over n_episodes FK rollouts."""
    rng = np.random.default_rng(seed_offset)
    rewards = [
        fk_rollout(
            make_env_seeded, actor_critic, k, rng,
            base_seed=seed_offset + i * 1000,
            independent_actions=INDEPENDENT_ACTIONS,
            temperature=temperature,
        )
        for i in range(n_episodes)
    ]
    return rewards


def _summarise(rewards, label) -> dict[str, Any]:
    mean = statistics.mean(rewards)
    std  = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    log.info("%-34s  mean=%.4f  std=%.4f  n=%d", label, mean, std, len(rewards))
    return {"label": label, "mean": mean, "std": std, "n": len(rewards), "rewards": rewards}


def main():
    log.info("FK vs PPO — offline §7.3 comparison  (N_TRAIN=%d N_EVAL=%d FK_KS=%s)",
             N_TRAIN, N_EVAL, FK_KS)

    records = load_records()
    windows = CommitmentWindowDetector().detect(records)
    targets = list(config.STAGE3_TARGETS)
    cw      = {t: windows[t] for t in targets if t in windows}
    seed    = int(os.environ.get("SEED", 42))

    # Training env uses intermediate reward shaping (helps the value function, §6); the FK
    # eval env uses intermediate_reward_scale=0.0 so a particle's terminal-step reward is
    # exactly its terminal compute_reward (and k=1 == a single greedy/random episode).
    train_env = DiffusionInterventionEnv(
        targets=targets, commitment_windows=cw, oracle_mode="offline",
        intermediate_reward_scale=float(os.environ.get("INTERMEDIATE_REWARD_SCALE", 0.1)),
        seed=seed,
    )

    def make_env_seeded(s: int):
        return DiffusionInterventionEnv(
            targets=targets, commitment_windows=cw, oracle_mode="offline",
            intermediate_reward_scale=0.0, seed=s,
        )

    # --- Train PPO ---
    log.info("Training PPO: %d episodes, %d steps/episode ...", N_TRAIN, N_WINDOW_STEPS)
    actor_critic = ActorCritic.build(
        obs_dim=train_env.observation_space.shape[0],
        n_actions=train_env.n_actions,
        hidden=(64, 64),
    )
    ppo_cfg = PPOFinetuneConfig(
        total_episodes=N_TRAIN, ppo_update_freq=32, n_epochs=4, batch_size=32,
        learning_rate=3e-4, clip_range=float(os.environ.get("CLIP_RANGE", 0.1)),
        save_every=N_TRAIN + 1,
        save_dir=str(config.EXPERIMENTS_LOG_DIR / "fk_vs_ppo_offline" / "checkpoints"),
    )
    train_log = train_lora_ppo(train_env, actor_critic, ppo_cfg, verbose=True)
    log.info("Training done. mean_reward=%.4f", train_log["mean_reward"])

    # --- Evaluate the 2×2 across the k sweep ---
    # eval seeds disjoint from training seeds via seed_offset.
    off = N_TRAIN
    results: dict[str, Any] = {"by_k": {}, "config": {
        "n_train": N_TRAIN, "n_eval": N_EVAL, "fk_ks": list(FK_KS),
        "temperature": FK_TEMPERATURE, "independent_actions": INDEPENDENT_ACTIONS,
    }}
    for k in FK_KS:
        ppo_r    = _eval_fk(make_env_seeded, actor_critic, k, N_EVAL, off, FK_TEMPERATURE)
        random_r = _eval_fk(make_env_seeded, None,         k, N_EVAL, off, FK_TEMPERATURE)
        results["by_k"][k] = {
            "ppo":    _summarise(ppo_r,    f"PPO      (k={k})"),
            "random": _summarise(random_r, f"random   (k={k})"),
        }

    out_dir = config.EXPERIMENTS_LOG_DIR / "fk_vs_ppo_offline"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # --- 2×2 table (no-FK = smallest k, +FK = largest k) ---
    k_nofk = min(FK_KS)
    k_fk   = max(FK_KS)
    cell = lambda pol, k: results["by_k"][k][pol]["mean"]
    c1 = cell("random", k_nofk)   # floor
    c2 = cell("ppo",    k_nofk)   # Stage 1-3
    c3 = cell("random", k_fk)     # search-only
    c4 = cell("ppo",    k_fk)     # policy + search

    print(f"\n{'='*60}")
    print(f"  FK Steering vs PPO — §7.3 2×2  (offline, {N_WINDOW_STEPS}-step MDP)")
    print(f"  N_eval={N_EVAL}  |  no-FK k={k_nofk}  |  +FK k={k_fk}  "
          f"|  {'independent' if INDEPENDENT_ACTIONS else 'shared'}-action")
    print(f"{'='*60}")
    print(f"  {'':<16}{'No FK (k=%d)' % k_nofk:>16}{'+FK (k=%d)' % k_fk:>16}")
    print(f"  {'Random cond.':<16}{c1:>16.4f}{c3:>16.4f}")
    print(f"  {'PPO policy':<16}{c2:>16.4f}{c4:>16.4f}")
    print(f"{'='*60}")
    print(f"  ordering:  4={c4:.4f}  2={c2:.4f}  3={c3:.4f}  1={c1:.4f}")
    if c4 > c2 > c3 > c1:
        print("  -> 4>2>3>1: policy adds value beyond search; complementary.")
    elif abs(c3 - c2) < 0.01:
        print("  -> 3≈2: learning and search solve the same problem (compute-efficiency story).")
    else:
        print("  -> mixed ordering; see results.json.")
    print(f"  FK gain (PPO):     k={k_fk} vs k={k_nofk}:  {c4 - c2:+.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
