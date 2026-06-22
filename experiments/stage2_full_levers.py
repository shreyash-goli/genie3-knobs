"""Stage 2 -- full three-lever discrete action space (timestep x length x hotspot).

Re-runs the same fixed / random / bandit / PPO comparison on the factored action space.

Data caveat (surfaced honestly): the logged sweeps only varied the hotspot/length levers
at a single timestep (the hotspot_ablation sweep at branch_t_800), so most of the
timestep x length x hotspot grid is unpopulated.  The OfflineRewardModel fills those cells
by documented back-off (see oracle/reward_oracle.py), and the fraction of evaluation
episodes that hit each back-off level is logged per episode -- so the write-up can state how
much of Stage 2 is grounded in exact-cell data vs. interpolation.

Because the hotspot/length variants only exist for a subset of targets, we restrict to the
targets that actually have those variants (otherwise those lever values would collapse).
"""

from __future__ import annotations

import sys

from experiments._harness import StageConfig, run_stage
from oracle.reward_oracle import OfflineRewardModel


def _targets_with_lever_variation() -> list[str]:
    """Targets that have >1 hotspot mode OR >1 length delta in the data (so the extra
    levers are meaningful)."""
    oracle = OfflineRewardModel()
    keep = []
    for t in oracle.targets():
        if len(oracle.available_modes(t)) > 1 or len(oracle.available_length_deltas(t)) > 1:
            keep.append(t)
    return keep or oracle.targets()


def main() -> int:
    targets = _targets_with_lever_variation()
    print(f"[stage2] targets with lever variation: {targets}")
    stage = StageConfig(
        name="stage2_full_levers",
        levers=("timestep", "length", "hotspot"),
        targets=targets,
        source_sweep=None,  # use all sweeps so the hotspot/length cells exist
        seeds=(0, 1, 2),
        train_budget=8000,
        ppo_timesteps=30_000,
        eval_per_target=200,
    )
    summary = run_stage(stage)
    print("\n[stage2] full-lever comparison complete; see "
          "data/experiment_logs/stage2_full_levers/summary.json")
    return 0 if summary["ppo_beats_bandit"] else 0  # informational, not a gate


if __name__ == "__main__":
    sys.exit(main())
