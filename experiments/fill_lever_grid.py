"""Fill missing lever cells for BHRF1 and InsulinR via the live oracle.

The offline dataset only has hotspot/length variation at branch_t_800.
This script runs the live oracle across the full grid:
  6 timesteps × 3 hotspot modes × 2 length variants × 2 targets = 72 calls

Each call generates RLKNOBS_NUM_CHILDREN children (default 5), evaluates them
with ProteinMPNN + ColabFold, and appends results to data/records.jsonl.

Usage (GPU node, genie3 conda env):
    conda activate genie3
    export RLKNOBS_GENIE3_ROOT=~/genie3
    export RLKNOBS_LIVE_SCRATCH=/pscratch/sd/s/shreyash/rlknobs_live
    export RLKNOBS_KEEP_LIVE_SCRATCH=1   # optional: keep PDBs for hotspot coverage later
    python -m experiments.fill_lever_grid [--targets 01_bhrf1 06_insulinr] [--dry-run]

Skip cells that already exist in records.jsonl (resume-safe).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import config
from instrumentation.trajectory_logger import load_records
from oracle.live_oracle import LiveRewardModel
from oracle.reward_oracle import compute_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# length_delta=30 is interpolated in the offline sim; for the live oracle we only
# generate {0, 60} since +30 has no distinct Genie3 problem variant.
LIVE_LENGTH_DELTAS = (0, 60)


def _already_logged(
    records: list[dict],
    target: str,
    timestep: int,
    hotspot_mode: str,
    length_delta: int,
) -> bool:
    """True if at least one live oracle record already exists for this cell."""
    for r in records:
        if (
            r.get("target") == target
            and r.get("branch_timestep") == timestep
            and r.get("hotspot_mode") == hotspot_mode
            and r.get("length_delta") == length_delta
            and r.get("source", "") == "live_oracle"
        ):
            return True
    return False


def _append_record(record: dict) -> None:
    config.ensure_data_dirs()
    with open(config.DATASET_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")


def build_grid(targets: list[str]) -> list[tuple]:
    """All (target, timestep, hotspot_mode, length_delta) combos to fill."""
    cells = []
    for target in targets:
        for ts in config.BRANCH_TIMESTEPS:
            for mode in config.HOTSPOT_MODES:
                for ld in LIVE_LENGTH_DELTAS:
                    cells.append((target, ts, mode, ld))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill lever grid via live oracle")
    parser.add_argument(
        "--targets", nargs="+",
        default=list(config.STAGE3_TARGETS),
        help="Target IDs to fill (default: BHRF1 + InsulinR contrast pair)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the cells that would be run without calling the oracle",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip cells already present in records.jsonl (default: True)",
    )
    args = parser.parse_args()

    existing = load_records() if args.skip_existing else []
    grid = build_grid(args.targets)

    pending = [
        (t, ts, mode, ld) for (t, ts, mode, ld) in grid
        if not (args.skip_existing and _already_logged(existing, t, ts, mode, ld))
    ]

    log.info(
        "Grid: %d total cells, %d pending (%d already logged)",
        len(grid), len(pending), len(grid) - len(pending),
    )

    if args.dry_run:
        for cell in pending:
            print(f"  target={cell[0]}  ts={cell[1]}  mode={cell[2]}  length_delta={cell[3]}")
        print(f"\n{len(pending)} cells would be run.")
        return

    oracle = LiveRewardModel()
    n_ok = 0
    n_err = 0

    for i, (target, timestep, hotspot_mode, length_delta) in enumerate(pending):
        log.info(
            "[%d/%d] target=%s ts=%d mode=%s len=%d",
            i + 1, len(pending), target, timestep, hotspot_mode, length_delta,
        )
        t0 = time.time()
        try:
            metrics, backoff = oracle.sample(
                target=target,
                timestep=timestep,
                hotspot_mode=hotspot_mode,
                length_delta=length_delta,
            )
            elapsed = time.time() - t0

            if "error" in metrics:
                log.warning("Oracle error for cell: %s", metrics["error"])
                n_err += 1
                continue

            reward = compute_reward(metrics)
            record = {
                **{k: v for k, v in metrics.items() if k != "x_T"},  # x_T is numpy, not JSON
                "target": target,
                "branch_timestep": timestep,
                "hotspot_mode": hotspot_mode,
                "length_delta": length_delta,
                "reward": reward,
                "backoff": backoff,
                "source": "live_oracle",
                "elapsed_s": round(elapsed, 1),
            }
            _append_record(record)
            log.info(
                "  -> iptm=%.3f  reward=%.4f  elapsed=%.0fs",
                metrics.get("iptm", 0.0), reward, elapsed,
            )
            n_ok += 1

        except Exception as e:
            log.error("Unhandled error for %s ts=%d: %s", target, timestep, e)
            n_err += 1

    log.info("Done. %d succeeded, %d errors out of %d cells.", n_ok, n_err, len(pending))


if __name__ == "__main__":
    main()
