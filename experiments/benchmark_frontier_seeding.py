"""Benchmark frontier buffer seeding vs cold-start rollouts (offline oracle, no GPU).

Shows whether replaying high-reward lever combinations from the FrontierBuffer
improves mean reward over cold-start random exploration.

Protocol:
  1. 200 cold-start episodes (no buffer) → collect (dummy x_T, lever, reward)
  2. 200 warm-start episodes (sample lever from buffer with p_frontier=0.5)
  3. Compare mean reward cold vs warm

x_T tensors are Gaussian placeholders (the real tensors come from live Genie3 runs).
The novelty gate and reward-weighted sampling still exercise the full buffer logic.

Writes: data/experiment_logs/frontier_seeding/summary.json
"""

from __future__ import annotations

import json
import logging

import numpy as np

import config
from buffer.frontier_buffer import FrontierBuffer, FrontierEntry
from envs.genie_branch_env import GenieBranchEnv
from oracle.reward_oracle import OfflineRewardModel, compute_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DUMMY_SHAPE = (150, 3)   # placeholder x_T shape (150 Cα atoms, 3 coords)
_N_COLD = 200
_N_WARM = 200
_P_FRONTIER = 0.5


def _run_episodes(env: GenieBranchEnv, oracle: OfflineRewardModel,
                  n: int, buf: FrontierBuffer, use_buf: bool, rng) -> list[float]:
    """Run n episodes, optionally seeding from frontier buffer."""
    rewards = []
    for _ in range(n):
        obs, info = env.reset()
        target = info["target"]

        # pick action: from buffer's stored lever combo or random
        if use_buf and rng.random() < _P_FRONTIER:
            entry = buf.top(target)
            if entry is not None:
                action = env.encode_levers(entry.levers)
                if action is None:
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()
        else:
            action = env.action_space.sample()

        _, reward, _, _, step_info = env.step(action)
        rewards.append(float(reward))

        # add to buffer with dummy x_T
        x_T = rng.standard_normal(_DUMMY_SHAPE).astype(np.float32)
        levers = step_info.get("action", {})
        if levers:
            entry = FrontierEntry(
                x_T=x_T,
                target=target,
                levers=levers,
                reward=float(reward),
                metrics=step_info,
            )
            buf.update(entry)
    return rewards


def main():
    log_dir = config.EXPERIMENTS_LOG_DIR / "frontier_seeding"
    log_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    oracle = OfflineRewardModel()
    env = GenieBranchEnv(
        oracle=oracle,
        levers=("timestep", "length", "hotspot"),
        seed=42,
    )

    buf = FrontierBuffer(size=32, epsilon=0.1, temperature=1.0, p_frontier=_P_FRONTIER, seed=42)
    buf.initialize(targets=env.targets)

    log.info("Phase 1: cold-start (%d episodes)", _N_COLD)
    cold_rewards = _run_episodes(env, oracle, _N_COLD, buf, use_buf=False, rng=rng)
    log.info("cold  mean=%.4f  buf_size=%d", np.mean(cold_rewards), len(buf))

    log.info("Phase 2: warm-start from buffer (%d episodes, p_frontier=%.1f)", _N_WARM, _P_FRONTIER)
    warm_rewards = _run_episodes(env, oracle, _N_WARM, buf, use_buf=True, rng=rng)
    log.info("warm  mean=%.4f  buf_size=%d", np.mean(warm_rewards), len(buf))

    summary = {
        "cold_mean_reward": float(np.mean(cold_rewards)),
        "cold_std_reward": float(np.std(cold_rewards)),
        "warm_mean_reward": float(np.mean(warm_rewards)),
        "warm_std_reward": float(np.std(warm_rewards)),
        "delta": float(np.mean(warm_rewards) - np.mean(cold_rewards)),
        "buffer_stats": buf.stats(),
        "n_cold": _N_COLD,
        "n_warm": _N_WARM,
        "p_frontier": _P_FRONTIER,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Frontier Buffer Seeding ===")
    print(f"  cold  mean_reward = {summary['cold_mean_reward']:.4f} ± {summary['cold_std_reward']:.4f}")
    print(f"  warm  mean_reward = {summary['warm_mean_reward']:.4f} ± {summary['warm_std_reward']:.4f}")
    print(f"  delta = {summary['delta']:+.4f}")
    print(f"  buffer: {len(buf)} entries across {len(env.targets)} targets")
    print(f"  logs: {log_dir}/summary.json")


if __name__ == "__main__":
    main()
