"""Stage 1 -- minimum viable demo: single-dimension (branch-timestep) action space.

Action = choose 1 of the available branch timesteps (length/hotspot levers dropped).
Reward = the standard terminal reward (success-dominated).  Compares fixed / random /
bandit / PPO.

Stopping rule (Section 2.2): if PPO cannot beat the contextual bandit here, stop and
diagnose before adding lever dimensions.  This script prints the verdict and exits non-zero
when PPO fails to beat the bandit, so it can gate Stage 2 in a pipeline.

We restrict to the trajectory_branching_v2 sweep: a clean single-variable timestep sweep
(6 timesteps x 10 targets x 40 children), the right data slice for a timestep-only study.
"""

from __future__ import annotations

import sys

from experiments._harness import StageConfig, run_stage


def main() -> int:
    stage = StageConfig(
        name="stage1_timestep_only",
        levers=("timestep",),
        source_sweep="trajectory_branching_v2",
        seeds=(0, 1, 2),
        train_budget=4000,
        ppo_timesteps=20_000,
        eval_per_target=200,
    )
    summary = run_stage(stage)
    if not summary["ppo_beats_bandit"]:
        print("\n[stage1] STOPPING RULE: PPO did not beat the bandit on the 1-D action "
              "space. Diagnose before proceeding to Stage 2.")
        return 1
    print("\n[stage1] PPO beat the bandit -> Stage 2 (full levers) is warranted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
