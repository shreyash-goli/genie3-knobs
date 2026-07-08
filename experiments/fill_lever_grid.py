"""Fill missing lever cells for BHRF1 and InsulinR via the live oracle.

The offline dataset only has hotspot/length variation at branch_t_800.
This script iterates the full 6 timesteps × 3 hotspot modes × 2 length variants ×
2 targets = 72-cell grid, but only dispatches cells that have a backing genie3
problem JSON (see oracle.live_oracle._build_selection): combined hotspot+length
variants and InsulinR's missed_only don't exist yet and are skipped up front rather
than discovered via a wasted oracle call. Of the 72 cells, 25 are currently fillable
after accounting for what's already logged and what has no dataset variant.

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
from concurrent.futures import as_completed
from pathlib import Path

import config
from instrumentation.trajectory_logger import load_records
from oracle.live_oracle import (
    LiveRewardModel,
    MultiGPULiveRewardModel,
    NoDatasetVariant,
    _build_selection,
)
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
    """True if a record already exists for this exact cell, from any source.

    The original offline sweep already covers some cells (e.g. `(all, 0)` at every
    timestep, and a few extra combos at branch_t=800 -- see the module docstring). Those
    records are tagged with `source_sweep` values like "trajectory_branching_v2", not
    `source == "live_oracle"`. Matching on `source == "live_oracle"` alone (as this used to)
    never recognizes that existing coverage, so every run would regenerate cells the offline
    dataset already has real data for -- wasted ~10 min/cell of live-oracle compute.
    """
    for r in records:
        if (
            r.get("target") == target
            and r.get("branch_timestep") == timestep
            and r.get("hotspot_mode") == hotspot_mode
            and r.get("length_delta") == length_delta
        ):
            return True
    return False


def _append_record(record: dict, output_path: Path) -> None:
    config.ensure_data_dirs()
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _process_result(
    target: str, timestep: int, hotspot_mode: str, length_delta: int,
    metrics: dict, backoff: int, elapsed: float, output_path: Path,
) -> bool:
    """Build a record from one oracle result, append it, log it. Returns True on success."""
    if "error" in metrics:
        log.warning(
            "Oracle error for target=%s ts=%d mode=%s len=%d: %s",
            target, timestep, hotspot_mode, length_delta, metrics["error"],
        )
        return False

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
    _append_record(record, output_path)
    log.info(
        "  -> target=%s ts=%d mode=%s len=%d  iptm=%.3f  reward=%.4f  elapsed=%.0fs",
        target, timestep, hotspot_mode, length_delta, metrics.get("iptm", 0.0), reward, elapsed,
    )
    return True


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
    parser.add_argument(
        "--output", type=Path, default=config.DATASET_JSONL,
        help=(
            "Where to append new records (default: data/records.jsonl). Point separate "
            "concurrent jobs at separate files -- data/records.jsonl lives on NFS, where "
            "concurrent O_APPEND writes from two different compute nodes are not guaranteed "
            "atomic and can corrupt lines. Merge the per-job files into records.jsonl once "
            "all jobs finish."
        ),
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help=(
            "Number of GPUs to dispatch across concurrently within this job (default: 1, "
            "sequential -- unchanged behavior). Requires the job to actually have this many "
            "GPUs allocated (e.g. --gpus=N in SLURM); cells are independent, so this "
            "parallelizes close to linearly. Each device processes its share of cells "
            "sequentially via MultiGPULiveRewardModel; results are appended as they "
            "complete, in completion order rather than grid order."
        ),
    )
    args = parser.parse_args()

    existing = load_records() if args.skip_existing else []
    grid = build_grid(args.targets)

    not_logged = [
        (t, ts, mode, ld) for (t, ts, mode, ld) in grid
        if not (args.skip_existing and _already_logged(existing, t, ts, mode, ld))
    ]

    # Filter out cells with no backing genie3 problem JSON *before* dispatch, rather
    # than discovering it via an empty-dataloader StopIteration after paying the full
    # model-load cost. Only single-axis hotspot/length variants exist today -- no
    # combined variant (e.g. ablate_competitors + length+60), and InsulinR has no
    # missed_only variant at all. See NEXT_STEPS.md for the follow-up to generate these.
    pending = []
    no_variant = []
    for cell in not_logged:
        t, ts, mode, ld = cell
        try:
            _build_selection(t, mode, ld)
            pending.append(cell)
        except NoDatasetVariant as e:
            no_variant.append((cell, str(e)))

    log.info(
        "Grid: %d total cells, %d pending, %d have no dataset variant yet "
        "(%d already logged)  output=%s",
        len(grid), len(pending), len(no_variant),
        len(grid) - len(not_logged), args.output,
    )
    if no_variant:
        log.info("Skipping (no dataset variant):")
        for (t, ts, mode, ld), reason in no_variant:
            log.info("  target=%s ts=%d mode=%s len=%d -- %s", t, ts, mode, ld, reason)

    if args.dry_run:
        for cell in pending:
            print(f"  target={cell[0]}  ts={cell[1]}  mode={cell[2]}  length_delta={cell[3]}")
        print(f"\n{len(pending)} cells would be run.")
        return

    n_ok = 0
    n_err = 0

    if args.num_gpus <= 1:
        oracle = LiveRewardModel()
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
                ok = _process_result(
                    target, timestep, hotspot_mode, length_delta,
                    metrics, backoff, elapsed, args.output,
                )
                n_ok += int(ok)
                n_err += int(not ok)
            except Exception as e:
                log.error("Unhandled error for %s ts=%d: %s", target, timestep, e)
                n_err += 1
    else:
        log.info(
            "Dispatching %d cells across %d GPUs (MultiGPULiveRewardModel)...",
            len(pending), args.num_gpus,
        )
        multi = MultiGPULiveRewardModel(devices=list(range(args.num_gpus)))
        future_to_cell = {}
        submit_time = {}
        try:
            for i, cell in enumerate(pending):
                target, timestep, hotspot_mode, length_delta = cell
                fut = multi.submit(i, target, timestep, hotspot_mode, length_delta)
                future_to_cell[fut] = cell
                submit_time[fut] = time.time()

            for n_done, fut in enumerate(as_completed(future_to_cell), start=1):
                target, timestep, hotspot_mode, length_delta = future_to_cell[fut]
                # Elapsed here is wall time since this cell was handed to its device's
                # queue, which includes any queue-wait behind earlier cells on the same
                # device -- not pure oracle execution time. Fine as a rough progress
                # signal; don't read it as a precise per-cell cost like the sequential path.
                elapsed = time.time() - submit_time[fut]
                try:
                    metrics, backoff = fut.result()
                    ok = _process_result(
                        target, timestep, hotspot_mode, length_delta,
                        metrics, backoff, elapsed, args.output,
                    )
                    n_ok += int(ok)
                    n_err += int(not ok)
                except Exception as e:
                    log.error("Unhandled error for %s ts=%d: %s", target, timestep, e)
                    n_err += 1
                log.info("[%d/%d] complete", n_done, len(pending))
        finally:
            multi.shutdown()

    log.info("Done. %d succeeded, %d errors out of %d cells.", n_ok, n_err, len(pending))


if __name__ == "__main__":
    main()
