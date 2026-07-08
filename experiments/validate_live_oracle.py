"""Validate that LiveRewardModel produces rewards consistent with the offline dataset.

Runs 1 live oracle call per target (Genie3 → ProteinMPNN → ColabFold) and compares
the result to the offline cell mean for the same (target, timestep) lever cell.
Lightweight sanity check before running full PPO with the live oracle.

Writes: data/experiment_logs/live_oracle_validation/results.json

Prerequisites (GPU node, genie3 conda env):
    conda activate genie3
    export RLKNOBS_GENIE3_ROOT=~/genie3
    export RLKNOBS_LIVE_SCRATCH=/pscratch/sd/s/shreyash/rlknobs_live
    python -m experiments.validate_live_oracle
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import config
from oracle.live_oracle import LiveRewardModel
from oracle.reward_oracle import OfflineRewardModel, compute_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    log_dir = config.EXPERIMENTS_LOG_DIR / "live_oracle_validation"
    log_dir.mkdir(parents=True, exist_ok=True)

    offline = OfflineRewardModel()
    live = LiveRewardModel()

    targets = config.STAGE3_TARGETS  # BHRF1 + InsulinR (cheapest to validate)
    test_timestep = 800              # the one timestep with hotspot/length data
    results = []

    for target in targets:
        log.info("=== %s ===", target)

        # offline baseline (cell mean)
        offline_records = []
        for _ in range(20):
            m, _ = offline.sample(target, test_timestep, "all", 0)
            offline_records.append(compute_reward(m))
        offline_mean = sum(offline_records) / len(offline_records)

        # live oracle (single call, n_children from env var or default=5)
        try:
            live_metrics, _ = live.sample(target, test_timestep, "all", 0)
            live_reward = compute_reward(live_metrics)
            live_ok = True
            # x_T is a numpy array (frozen diffusion state) -- not JSON serializable;
            # strip it before writing results.json (same pattern as fill_lever_grid.py).
            live_metrics = {k: v for k, v in live_metrics.items() if k != "x_T"}
        except Exception as e:
            log.error("Live oracle failed for %s: %s", target, e)
            live_metrics = {"error": str(e)}
            live_reward = float("nan")
            live_ok = False

        result = {
            "target": target,
            "timestep": test_timestep,
            "offline_mean_reward": offline_mean,
            "live_reward": live_reward,
            "live_ok": live_ok,
            "live_metrics": live_metrics,
            "delta": live_reward - offline_mean if live_ok else None,
        }
        results.append(result)
        log.info(
            "%s  offline_mean=%.4f  live=%.4f  delta=%s",
            target, offline_mean, live_reward,
            f"{result['delta']:+.4f}" if result["delta"] is not None else "N/A",
        )

    (log_dir / "results.json").write_text(json.dumps(results, indent=2))
    log.info("Results written to %s", log_dir / "results.json")

    # exit 1 if any live call failed
    if not all(r["live_ok"] for r in results):
        log.error("Some live oracle calls failed — check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
