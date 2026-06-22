"""Compare fixed / random / bandit / PPO on the single-dimension branch-timestep action space.

Action = choose 1 of the available branch timesteps (length/hotspot levers fixed at defaults).
Reward = the standard terminal reward (success-dominated).

Stopping rule: if PPO cannot beat the contextual bandit here, the RL framing is not earning
its complexity on the 1-D problem — diagnose before expanding the action space.  Exits
non-zero when PPO fails so this can gate compare_full_levers.py in a pipeline.

Restricted to trajectory_branching_v2: a clean single-variable timestep sweep
(6 timesteps × 10 targets × 40 children), 100% exact-cell grounding.
"""

from __future__ import annotations

import sys

from experiments._harness import StageConfig, run_stage


def main() -> int:
    stage = StageConfig(
        name="timestep_lever",
        levers=("timestep",),
        source_sweep="trajectory_branching_v2",
        seeds=(0, 1, 2),
        train_budget=4000,
        ppo_timesteps=20_000,
        eval_per_target=200,
    )
    summary = run_stage(stage)
    if not summary["ppo_beats_bandit"]:
        print("\nSTOPPING RULE: PPO did not beat the bandit on the 1-D action space. "
              "Diagnose before expanding to full levers.")
        return 1
    print("\nPPO beat the bandit on timestep-only — full-lever comparison is warranted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
