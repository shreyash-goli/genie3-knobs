"""Compare fixed / random / bandit / PPO on the full three-lever action space (timestep × length × hotspot).

Data caveat: hotspot/length levers were only varied at a single timestep (branch_t_800 in
hotspot_ablation), so most of the timestep × length × hotspot grid is unpopulated.
OfflineRewardModel fills those cells via documented back-off; the fraction hitting each
back-off level is logged per episode in episodes.jsonl.

Restricted to targets that actually have hotspot/length variation in the data (otherwise
those lever values collapse to one value and the action space degenerates).
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
    print(f"targets with lever variation: {targets}")
    stage = StageConfig(
        name="full_levers",
        levers=("timestep", "length", "hotspot"),
        targets=targets,
        source_sweep=None,  # use all sweeps so the hotspot/length cells exist
        seeds=(0, 1, 2),
        train_budget=8000,
        ppo_timesteps=30_000,
        eval_per_target=200,
    )
    summary = run_stage(stage)
    print("\nFull-lever comparison complete; see "
          "data/experiment_logs/full_levers/summary.json")
    return 0 if summary["ppo_beats_bandit"] else 0  # informational, not a gate


if __name__ == "__main__":
    sys.exit(main())
