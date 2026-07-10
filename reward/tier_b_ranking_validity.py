"""Tier B -- reward ranking-validity test (the decisive, GPU-free experiment).

The reform's job is NOT to widen the PPO-vs-bandit margin (NEXT_STEPS.md §3.1/§6 showed that
margin lives inside the ±0.024 noise floor and is just cross-target specialisation). Its job
is to make the reward *valid*: rank a genuine binder above an "impostor" that scores well on
the geometry-blind global metrics (high iptm, low interface pAE) while contacting none of the
target's true hotspots -- the documented InsulinR failure mode.

That is a pure ranking property. We test it by re-scoring the kept live-oracle scratch designs
(hotspot_coverage computed retroactively, see reward/retro_metrics.py -- no new oracle calls)
under each candidate reward and asking three questions per design:

  1. Failure-mode separation. Among *designable* designs (pass the iptm/pAE gate), split into
     genuine (coverage >= HIGH) and impostor (coverage <= LOW). Report AUC = P(genuine ranks
     above impostor by reward). ~0.5 => the reward cannot tell them apart (expected for the
     current geometry-blind reward); ->1.0 => it correctly down-ranks impostors.
  2. Coverage sensitivity. Spearman(reward, hotspot_coverage) over all rows. ~0 for current;
     strongly positive if the reward actually responds to contact geometry.
  3. Ranking change vs current. Spearman(reward_design, reward_current). Near 1.0 => the
     reform is cosmetic (same ordering); lower => it genuinely re-orders designs.

Run:  conda activate genie3 && python -m reward.tier_b_ranking_validity
Env:  RLKNOBS_LIVE_SCRATCH overrides the scratch dir.

Caveat carried honestly: iCS is None in this data (no PAE sidecars persisted for these runs),
so the gated design falls back to global pAE for its geometry term here. The GATE
(hotspot_coverage) is real; the iCS refinement is validated separately (NEXT_STEPS.md §1.7)
and will only strengthen the separation once a PAE-persisting re-run populates it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import config
from reward.retro_metrics import DEFAULT_SCRATCH, collect_rows
from reward.reward_designs import REWARD_DESIGNS, _designable, design_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# bucket thresholds for the genuine/impostor split (on hotspot_coverage)
COVERAGE_HIGH = 0.75   # genuine: contacts most true hotspots
COVERAGE_LOW = 0.50    # impostor: misses at least half the true hotspots


def _ranks(x: list[float]):
    import numpy as np
    return np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)


def _spearman(x: list[float], y: list[float]) -> float:
    import numpy as np
    if len(x) < 3:
        return float("nan")
    xr, yr = _ranks(x), _ranks(y)
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _partial_spearman(y: list[float], x: list[float], controls: list[list[float]]) -> float:
    """Spearman(y, x) after linearly regressing rank(y) and rank(x) on the ranked controls.

    This is the discriminator that works even when the data has no designable impostors:
    it asks whether the reward ``y`` carries information about coverage ``x`` *beyond* what
    the quality controls (iptm, pae) already explain. ~0 => the reward is geometry-blind
    (all its coverage correlation is inherited from quality); >0 => it genuinely responds to
    contact geometry on top of quality."""
    import numpy as np
    if len(y) < 5:
        return float("nan")
    yr, xr = _ranks(y), _ranks(x)
    C = np.column_stack([_ranks(c) for c in controls] + [np.ones(len(y))])

    def resid(v):
        beta, *_ = np.linalg.lstsq(C, v, rcond=None)
        return v - C @ beta

    ry, rx = resid(yr), resid(xr)
    if ry.std() == 0 or rx.std() == 0:
        return float("nan")
    return float(np.corrcoef(ry, rx)[0, 1])


def _pearson(x: list[float], y: list[float]) -> float:
    import numpy as np
    if len(x) < 3:
        return float("nan")
    a, b = np.asarray(x, float), np.asarray(y, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _auc(pos: list[float], neg: list[float]) -> float:
    """P(a random pos scores above a random neg), ties counted as 0.5 (rank-based / MWU)."""
    import numpy as np
    if not pos or not neg:
        return float("nan")
    allv = np.array(pos + neg, dtype=float)
    ranks = np.argsort(np.argsort(allv)).astype(float) + 1.0  # 1-based
    # average-rank tie handling
    order = np.argsort(allv)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i:j + 1]]).mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    n_pos = len(pos)
    r_pos = ranks[:n_pos].sum()
    u = r_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * len(neg)))


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # score every row under every design (coverage is present on these rows)
    for r in rows:
        r["_rewards"] = {name: fn(r) for name, fn in REWARD_DESIGNS.items()}

    designable = [r for r in rows if _designable(r)]
    genuine = [r for r in designable if (r["hotspot_coverage"] or 0.0) >= COVERAGE_HIGH]
    impostor = [r for r in designable if (r["hotspot_coverage"] or 0.0) <= COVERAGE_LOW]

    # quality controls for the partial correlation (only rows where both exist)
    ctl_rows = [r for r in rows if r.get("iptm") is not None
                and r.get("avg_interface_pae") is not None
                and r.get("hotspot_coverage") is not None]
    iptm_c = [float(r["iptm"]) for r in ctl_rows]
    pae_c = [float(r["avg_interface_pae"]) for r in ctl_rows]
    cov_c = [float(r["hotspot_coverage"]) for r in ctl_rows]

    per_design: dict[str, Any] = {}
    for name in REWARD_DESIGNS:
        rew_all = [r["_rewards"][name] for r in rows]
        cov_all = [float(r["hotspot_coverage"] or 0.0) for r in rows]
        cur_all = [r["_rewards"]["current"] for r in rows]
        gpos = [r["_rewards"][name] for r in genuine]
        ineg = [r["_rewards"][name] for r in impostor]
        rew_c = [r["_rewards"][name] for r in ctl_rows]
        per_design[name] = {
            "separation_auc": _auc(gpos, ineg),
            "mean_reward_genuine": (sum(gpos) / len(gpos)) if gpos else float("nan"),
            "mean_reward_impostor": (sum(ineg) / len(ineg)) if ineg else float("nan"),
            "coverage_spearman": _spearman(rew_all, cov_all),
            "partial_coverage_spearman": _partial_spearman(rew_c, cov_c, [iptm_c, pae_c]),
            "vs_current_spearman": _spearman(rew_all, cur_all),
        }

    return {
        "n_rows": len(rows),
        "n_designable": len(designable),
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
        "coverage_high": COVERAGE_HIGH,
        "coverage_low": COVERAGE_LOW,
        # why the AUC test may be empty: does the failure mode even exist in this data?
        "n_designable_low_coverage_impostors": len(impostor),
        "n_good_pae_low_coverage_dissociations": sum(
            1 for r in rows
            if r.get("avg_interface_pae") is not None and r.get("hotspot_coverage") is not None
            and r["avg_interface_pae"] <= 10.0 and r["hotspot_coverage"] <= COVERAGE_LOW
        ),
        "pearson_pae_coverage": _pearson(pae_c, cov_c),
        "pearson_iptm_coverage": _pearson(iptm_c, cov_c),
        "per_design": per_design,
        "per_target": _per_target(rows),
    }


def _per_target(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tgt in sorted({r["target"] for r in rows}):
        trows = [r for r in rows if r["target"] == tgt]
        des = [r for r in trows if _designable(r)]
        gen = [r for r in des if (r["hotspot_coverage"] or 0.0) >= COVERAGE_HIGH]
        imp = [r for r in des if (r["hotspot_coverage"] or 0.0) <= COVERAGE_LOW]
        out[tgt] = {"n": len(trows), "n_designable": len(des),
                    "n_genuine": len(gen), "n_impostor": len(imp)}
        for name in REWARD_DESIGNS:
            out[tgt][f"auc_{name}"] = _auc([r["_rewards"][name] for r in gen],
                                           [r["_rewards"][name] for r in imp])
    return out


def _fmt(x: float) -> str:
    return "  n/a " if x != x else f"{x:+.3f}"  # x!=x is NaN


def main(scratch_dir: Optional[Path] = None) -> None:
    scratch_dir = scratch_dir or DEFAULT_SCRATCH
    log.info("collecting rows from %s ...", scratch_dir)
    rows = collect_rows(scratch_dir)
    if len(rows) < 10:
        log.error("only %d rows with predicted PDB + hotspot set; need >=10", len(rows))
        return
    result = analyse(rows)

    out_dir = config.EXPERIMENTS_LOG_DIR / "reward_tier_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    # strip the transient _rewards before dumping the summary
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))

    W = 74
    print("\n" + "=" * W)
    print("  Tier B -- reward ranking validity (kept scratch, no GPU)")
    print(f"  rows={result['n_rows']}  designable={result['n_designable']}  "
          f"genuine(cov>={COVERAGE_HIGH})={result['n_genuine']}  "
          f"impostor(cov<={COVERAGE_LOW})={result['n_impostor']}")
    print("=" * W)
    print("  Does the targeted failure mode even occur in this data?")
    print(f"    designable-but-low-coverage impostors : {result['n_designable_low_coverage_impostors']}")
    print(f"    good-pAE(<=10)-but-low-coverage        : {result['n_good_pae_low_coverage_dissociations']}")
    print(f"    Pearson(pae, coverage) = {_fmt(result['pearson_pae_coverage'])}   "
          f"Pearson(iptm, coverage) = {_fmt(result['pearson_iptm_coverage'])}")
    print("    (coverage co-varying with quality => the geometry-blind reward is not")
    print("     being exploited here; the AUC test below is empty when impostors=0.)")
    print("-" * W)
    print(f"  {'design':<18} {'sep AUC':>8} {'R|gen':>7} {'R|imp':>7} "
          f"{'cov ρ':>7} {'cov ρ|q':>8} {'vs cur':>7}")
    print("  " + "-" * (W - 2))
    for name, d in result["per_design"].items():
        print(f"  {name:<18} {_fmt(d['separation_auc']):>8} "
              f"{_fmt(d['mean_reward_genuine']):>7} {_fmt(d['mean_reward_impostor']):>7} "
              f"{_fmt(d['coverage_spearman']):>7} {_fmt(d['partial_coverage_spearman']):>8} "
              f"{_fmt(d['vs_current_spearman']):>7}")
    print("=" * W)
    print("  sep AUC : 0.5=can't separate genuine/impostor, 1.0=perfect (n/a if impostors=0)")
    print("  cov ρ   : Spearman(reward, coverage) -- raw responsiveness to geometry")
    print("  cov ρ|q : PARTIAL Spearman(reward, coverage | iptm, pae) -- geometry signal")
    print("            BEYOND quality. ~0 => geometry-blind; >0 => genuine new signal.")
    print("  vs cur  : Spearman(reward, current) -- 1.0 means the reform is cosmetic")
    print("\n  Per target (separation AUC):")
    print(f"  {'target':<14} {'n':>4} {'desig':>6} {'gen':>4} {'imp':>4} "
          + " ".join(f"{n[:8]:>9}" for n in REWARD_DESIGNS))
    for tgt, d in result["per_target"].items():
        print(f"  {tgt:<14} {d['n']:>4} {d['n_designable']:>6} {d['n_genuine']:>4} "
              f"{d['n_impostor']:>4} "
              + " ".join(f"{_fmt(d['auc_' + n]):>9}" for n in REWARD_DESIGNS))
    print("=" * W)
    print(f"  wrote {out_dir / 'results.json'}")
    print("=" * W + "\n")


if __name__ == "__main__":
    main()
