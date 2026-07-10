"""Tier A -- reward-form ablation at scale (all of records.jsonl, no GPU).

Complements Tier B. Tier B asks 'is the reform *valid*' on the small kept-scratch set that has
retroactive hotspot_coverage. Tier A asks two things the full 4075-record corpus *can* answer
without any geometry metric:

  1. Form stability. records.jsonl has no hotspot_coverage/iCS, so here the designs differ only
     by their FUNCTIONAL FORM on the metrics that are present (success/iptm/pae) -- i.e. flat
     average vs the signed gate. Does merely changing the form move the argmax lever cell per
     target (the thing offline PPO optimises toward)? If the argmax is stable, the reform is
     inert on existing training and safe to land; if it moves, a retrain is required. Reported
     as per-target argmax cell + Spearman of cell-mean-reward rankings.

  2. Developability reach. GRAVY and net charge ARE computable from binder_seq on every record.
     Report how many designs the §2.3 hard developability filter (GRAVY>0 or |charge|>10)
     would drop, and the reward gap between passing and failing designs -- i.e. how much signal
     a soft developability penalty would add on top of the structural reward.

Note: with coverage absent, the gated design runs ungated (coverage_missing='neutral') -- its
signed failure/efficiency behaviour is still exercised, but the coverage GATE is not (that is
Tier B). So Tier A is a form/at-scale safety and developability check, not a geometry test.

Run:  conda activate genie3 && python -m reward.tier_a_form_ablation
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

import config
from instrumentation.trajectory_logger import load_records
from oracle.live_oracle import _passes_developability
from reward.developability import DevelopabilityWeights, attach_panel
from reward.reward_designs import REWARD_DESIGNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _cell(r: dict[str, Any]) -> tuple:
    return (r["target"], r["branch_timestep"], r["hotspot_mode"], r["length_delta"])


def _spearman(x: list[float], y: list[float]) -> float:
    import numpy as np
    if len(x) < 3:
        return float("nan")
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def analyse(records: list[dict[str, Any]]) -> dict[str, Any]:
    # attach the full retroactive sequence developability panel (GRAVY / charge / pI /
    # instability -- all from binder_seq, no GPU) so the +dev design and the report can use it
    for r in records:
        attach_panel(r)

    # score every record under every design
    for r in records:
        r["_rewards"] = {name: fn(r) for name, fn in REWARD_DESIGNS.items()}

    # -- 1. form stability: per-target argmax lever cell under each design ----------------
    by_target_cell: dict[str, dict[tuple, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        for name in REWARD_DESIGNS:
            by_target_cell[r["target"]][_cell(r)][name].append(r["_rewards"][name])

    form: dict[str, Any] = {}
    for tgt, cells in by_target_cell.items():
        cell_means = {c: {n: sum(v) / len(v) for n, v in per.items()} for c, per in cells.items()}
        order = sorted(cell_means)
        argmax = {}
        rankvecs = {}
        for name in REWARD_DESIGNS:
            argmax[name] = max(cell_means, key=lambda c: cell_means[c][name])
            rankvecs[name] = [cell_means[c][name] for c in order]
        form[tgt] = {
            "n_cells": len(cells),
            "argmax_cell": {n: list(argmax[n]) for n in REWARD_DESIGNS},
            "argmax_matches_current": {
                n: (argmax[n] == argmax["current"]) for n in REWARD_DESIGNS},
            "cell_ranking_spearman_vs_current": {
                n: _spearman(rankvecs[n], rankvecs["current"]) for n in REWARD_DESIGNS},
        }

    # -- 2. developability reach ---------------------------------------------------------
    w = DevelopabilityWeights()
    seqrecs = [r for r in records if r.get("binder_seq")]
    passed, failed = [], []
    for r in seqrecs:
        ok, _, _ = _passes_developability(r["binder_seq"])  # hard filter: GRAVY/charge
        (passed if ok else failed).append(r)

    def _frac(pred) -> float:
        vals = [r for r in seqrecs if pred(r)]
        return len(vals) / len(seqrecs) if seqrecs else float("nan")

    dev = {
        "n_with_seq": len(seqrecs),
        "hard_filter": {  # GRAVY>0 or |charge|>10 -> dropped pre-oracle
            "n_fail": len(failed),
            "frac_fail": len(failed) / len(seqrecs) if seqrecs else float("nan"),
            "mean_current_reward_pass": _mean([r["_rewards"]["current"] for r in passed]),
            "mean_current_reward_fail": _mean([r["_rewards"]["current"] for r in failed]),
        },
        "soft_bands": {  # fraction out-of-band per soft term (BioPython-derived)
            "gravy_gt_0": _frac(lambda r: (r.get("gravy") or 0) > w.gravy_max),
            "abs_charge_gt_10": _frac(lambda r: abs(r.get("net_charge") or 0) > w.charge_abs_max),
            "pi_outside_6_9": _frac(
                lambda r: r.get("isoelectric_point") is not None
                and not (w.pi_low <= r["isoelectric_point"] <= w.pi_high)),
            "instability_gt_40": _frac(
                lambda r: r.get("instability_index") is not None
                and r["instability_index"] > w.instability_max),
        },
        # how much the +dev penalty moves reward vs the recommended design
        "mean_gated": _mean([r["_rewards"]["gated"] for r in seqrecs]),
        "mean_gated_dev": _mean([r["_rewards"]["gated+dev"] for r in seqrecs]),
    }

    return {"n_records": len(records), "form": form, "developability": dev}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _fmt(x: float) -> str:
    return " n/a " if x != x else f"{x:+.3f}"


def main() -> None:
    records = load_records()
    log.info("loaded %d records", len(records))
    result = analyse(records)

    out_dir = config.EXPERIMENTS_LOG_DIR / "reward_tier_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))

    W = 74
    print("\n" + "=" * W)
    print("  Tier A -- reward-form ablation at scale (records.jsonl, no geometry)")
    print(f"  records={result['n_records']}")
    print("=" * W)
    print("  1) Form stability -- does changing reward FORM move the argmax lever cell?")
    for tgt, d in result["form"].items():
        print(f"\n  {tgt}  ({d['n_cells']} lever cells)")
        print(f"    {'design':<18} {'argmax==current?':>16} {'cell-rank ρ vs cur':>19}")
        for name in REWARD_DESIGNS:
            print(f"    {name:<18} {str(d['argmax_matches_current'][name]):>16} "
                  f"{_fmt(d['cell_ranking_spearman_vs_current'][name]):>19}")
    print("\n" + "-" * W)
    dev = result["developability"]
    hf = dev["hard_filter"]
    sb = dev["soft_bands"]
    print(f"  2) Developability reach (all from binder_seq, no GPU; n={dev['n_with_seq']})")
    print(f"    HARD filter (GRAVY>0 or |charge|>10): drops {hf['n_fail']} "
          f"({hf['frac_fail']:.1%})")
    print(f"      mean current reward  pass={_fmt(hf['mean_current_reward_pass'])}  "
          f"fail={_fmt(hf['mean_current_reward_fail'])}  "
          f"(fail>=pass => reward prefers undevelopable)")
    print("    SOFT bands out-of-range:")
    print(f"      GRAVY>0 {sb['gravy_gt_0']:.1%}   |charge|>10 {sb['abs_charge_gt_10']:.1%}   "
          f"pI∉[6,9] {sb['pi_outside_6_9']:.1%}   instability>40 {sb['instability_gt_40']:.1%}")
    print(f"    mean recommended reward: gated={_fmt(dev['mean_gated'])}  "
          f"gated+dev={_fmt(dev['mean_gated_dev'])}  "
          f"(Δ={_fmt(dev['mean_gated_dev'] - dev['mean_gated'])})")
    print("=" * W)
    print(f"  wrote {out_dir / 'results.json'}")
    print("=" * W + "\n")


if __name__ == "__main__":
    main()
