"""Window-start ablation — isolates whether window_start_sweep's PPO-losing result comes
from forcing two *different* targets onto a shared window, or is a real property of the
tested window placements themselves.

Background (NEXT_STEPS.md section 3.1): after fixing the train_lora_ppo single-step-episode
bug, ppo_vs_bandit_offline.py (each target trained with its own auto-detected commitment
window) shows a clean PPO win. window_start_sweep.py (both targets forced onto the *same*
window_start/window_end override) still shows fixed beating PPO at all 5 tested placements,
by roughly the same margin as before the bug fix. Those two experiments differ in more than
one way (shared vs. per-target window, training budget, one joint policy vs. one per
placement), so it isn't yet clear which difference explains the gap.

This script reruns window_start_sweep's *exact* design (same window_end=700, same
window_start candidates, same run_placement() training/eval code) but one target at a time,
removing the shared-window-across-differing-targets confound while holding everything else
fixed. If PPO now wins (or is much closer) per-target, that confirms cross-target window
sharing was the dominant effect. If PPO still loses per-target by a similar margin, the
window placements themselves (750-950, decreasing schedule) are the harder regime,
independent of sharing.

Usage:
    conda run -n genie3 python -m experiments.window_start_ablation
"""

from __future__ import annotations

import json
import logging

import config
from envs.commitment_window import DiffusionInterventionEnv
from experiments.window_start_sweep import (
    N_EVAL,
    N_TRAIN,
    WINDOW_END_FIXED,
    WINDOW_START_CANDIDATES,
    run_placement,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    targets_all = list(config.STAGE3_TARGETS)
    log.info(
        "Window-start ablation (single target at a time)  N_TRAIN=%d  N_EVAL=%d  targets=%s",
        N_TRAIN, N_EVAL, targets_all,
    )

    all_results: dict[str, list[dict]] = {}
    for target in targets_all:
        probe_env = DiffusionInterventionEnv(targets=[target], oracle_mode="offline")
        obs_dim = probe_env.observation_space.shape[0]
        n_actions = probe_env.n_actions

        target_results = []
        for ws in WINDOW_START_CANDIDATES:
            result = run_placement(ws, [target], {}, obs_dim, n_actions)
            target_results.append(result)
        all_results[target] = target_results

    out_dir = config.EXPERIMENTS_LOG_DIR / "window_start_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "n_train": N_TRAIN,
            "n_eval": N_EVAL,
            "window_end_fixed": WINDOW_END_FIXED,
            "window_start_candidates": WINDOW_START_CANDIDATES,
            "targets": targets_all,
        },
        "results": all_results,
    }
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    log.info("Results written to %s/results.json", out_dir)

    print(f"\n{'=' * 72}")
    print("  Window-start ablation -- single target at a time")
    print(f"  N_train={N_TRAIN}  N_eval={N_EVAL}  window_end fixed={WINDOW_END_FIXED}")
    print(f"{'=' * 72}")
    joint_deltas = []
    for target, results in all_results.items():
        print(f"\n  target = {target}")
        print(f"  {'window_start':>12}  {'PPO mean':>10}  {'fixed best':>10}  {'delta':>8}")
        print(f"  {'-' * 50}")
        n_wins = 0
        for r in results:
            marker = ""
            if r["delta_ppo_vs_fixed"] > 0:
                marker = "  <-- PPO wins"
                n_wins += 1
            joint_deltas.append(r["delta_ppo_vs_fixed"])
            print(
                f"  {r['window_start']:>12}  {r['ppo_mean']:>10.4f}  "
                f"{r['fixed_best_mean']:>10.4f}  {r['delta_ppo_vs_fixed']:>+8.4f}{marker}"
            )
        print(f"  PPO beat fixed at {n_wins}/{len(results)} placements for {target}")
    print(f"\n  mean delta across all target x placement combos: "
          f"{sum(joint_deltas) / len(joint_deltas):+.4f}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
