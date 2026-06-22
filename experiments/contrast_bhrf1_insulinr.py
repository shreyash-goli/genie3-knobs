"""Target-dependent behaviour check: BHRF1 vs InsulinR.

Targets:
  * BHRF1     (01_bhrf1)   — longer binders helped recover missed hotspots in prior ablations.
  * InsulinR  (06_insulinr)— length did NOT generalise (B59/B91 resisted length interventions).

The key question for a learned policy (vs. a fixed heuristic): does it extend binder length
on BHRF1 but not on InsulinR?  Runs the full-lever comparison on just these two targets and
inspects each policy's per-target greedy action, writing a contrast.json verdict.
"""

from __future__ import annotations

import json
import sys

import config
from experiments._harness import StageConfig, run_stage


def _length_choice(levers: dict) -> int:
    return int(levers.get("length_delta", 0))


def main() -> int:
    targets = [config.STAGE3_LENGTH_HELPED, config.STAGE3_LENGTH_DID_NOT_HELP]
    stage = StageConfig(
        name="bhrf1_insulinr_contrast",
        levers=("timestep", "length", "hotspot"),
        targets=targets,
        source_sweep=None,
        seeds=(0, 1, 2),
        train_budget=6000,
        ppo_timesteps=30_000,
        eval_per_target=300,
    )
    summary = run_stage(stage)

    bandit_table = summary["detail_last_seed"].get("bandit_policy_table", {})
    ppo_table = summary["detail_last_seed"].get("ppo_policy_table", {})

    print("\n=== BHRF1 vs InsulinR: target-dependent behaviour check ===")
    print(f"  length-helped target     : {config.STAGE3_LENGTH_HELPED}")
    print(f"  length-didn't-help target: {config.STAGE3_LENGTH_DID_NOT_HELP}")
    for name, table in (("bandit", bandit_table), ("ppo", ppo_table)):
        print(f"\n  [{name}] per-target greedy action:")
        for t, levers in table.items():
            print(f"    {t}: {levers}")
        helped = _length_choice(table.get(config.STAGE3_LENGTH_HELPED, {}))
        not_helped = _length_choice(table.get(config.STAGE3_LENGTH_DID_NOT_HELP, {}))
        target_dependent = (table.get(config.STAGE3_LENGTH_HELPED)
                            != table.get(config.STAGE3_LENGTH_DID_NOT_HELP))
        extends_correctly = helped > not_helped
        print(f"    -> target-dependent action: {target_dependent} "
              f"(length: helped={helped}, not_helped={not_helped}; "
              f"extends-more-on-length-helped={extends_correctly})")

    # persist the contrast verdict alongside the summary
    log_dir = config.EXPERIMENTS_LOG_DIR / stage.name
    (log_dir / "contrast.json").write_text(json.dumps({
        "bandit_policy_table": bandit_table,
        "ppo_policy_table": ppo_table,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
