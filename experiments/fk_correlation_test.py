"""FK-Steering correlation test (NEXT_STEPS.md §7.1).

FK steering scores partial trajectories at intermediate timesteps using a cheap potential,
then resamples. For that to work, the cheap potential must correlate with the final terminal
reward. This measures that correlation for the two zero-oracle-cost geometric candidates from
§2.3 -- rg (radius of gyration of the binder) and nc_termini (N-to-C terminal distance) --
computed on the Genie3-generated binder, against the ColabFold-based terminal reward.

Data source: the live-oracle scratch dirs kept via RLKNOBS_KEEP_LIVE_SCRATCH=1 (the grid
fill + validation runs). Each run branched at some timestep t and ran children to completion,
so we have, per child: the generated complex PDB (child_N.pdb, binder = chain A) and the
terminal metrics (child_N_metrics.json). Correlation is reported overall and stratified by
branch timestep t, giving the "correlation vs. how far into the trajectory" curve §7.1 asks
for.

IMPORTANT CAVEAT (reported, not hidden): the geometric metric here is computed on the *final*
generated structure for each branch, not a true mid-trajectory x̂_t. Branch timestep t
parametrizes trajectory position (how much history was shared before branching), so the
t-stratified curve is a valid proxy, but the strict FK setup (score x̂_t at the branch point,
before full denoising) would need Genie3 denoiser hooks to emit intermediate x̂_0 predictions.
This test tells you whether the cheap metrics are predictive at all before that lift.

Usage:
    conda run -n genie3 python -m experiments.fk_correlation_test
    RLKNOBS_LIVE_SCRATCH=/pscratch/sd/s/shreyash/rlknobs_live \
        conda run -n genie3 python -m experiments.fk_correlation_test
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

import config
from oracle.reward_oracle import compute_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_BRANCH_RE = re.compile(r"branch_t_(\d+)")


def _binder_ca_coords(pdb_path: Path, binder_chain: str = "A") -> Optional[np.ndarray]:
    """Cα coordinates of the binder chain, in residue order. None if the chain is absent."""
    by_res: dict[int, np.ndarray] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA" or line[21] != binder_chain:
                continue
            resnum = int(line[22:26])
            by_res[resnum] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
    if not by_res:
        return None
    return np.array([by_res[r] for r in sorted(by_res)])


def _radius_of_gyration(ca: np.ndarray) -> float:
    """Rg of the binder Cα cloud. Larger = more extended/floppy for a given length."""
    center = ca.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((ca - center) ** 2, axis=1))))


def _nc_termini_distance(ca: np.ndarray) -> float:
    """Distance between the binder's N- and C-terminal Cα atoms."""
    return float(np.linalg.norm(ca[0] - ca[-1]))


def collect(scratch_dir: Path) -> list[dict]:
    """Walk the scratch dirs; per child return (branch_t, rg, nc_termini, terminal_reward)."""
    rows = []
    for metrics_path in scratch_dir.glob("run_*/branch_t_*/*/child_*_metrics.json"):
        m = _BRANCH_RE.search(str(metrics_path))
        if not m:
            continue
        branch_t = int(m.group(1))
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            continue
        if metrics.get("iptm") is None or "error" in metrics:
            continue
        child_id = metrics.get("child_id")
        pdb_path = metrics_path.parent / f"child_{child_id}.pdb"
        if not pdb_path.exists():
            continue
        ca = _binder_ca_coords(pdb_path)
        if ca is None or len(ca) < 2:
            continue
        rows.append({
            "branch_t": branch_t,
            "rg": _radius_of_gyration(ca),
            "nc_termini": _nc_termini_distance(ca),
            "terminal_reward": compute_reward(metrics),
        })
    return rows


def _pearson(x, y) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y) -> float:
    if len(x) < 3:
        return float("nan")
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    return _pearson(xr, yr)


def main():
    scratch_dir = Path(
        os.environ.get("RLKNOBS_LIVE_SCRATCH", "/pscratch/sd/s/shreyash/rlknobs_live")
    )
    log.info("Collecting from %s ...", scratch_dir)
    rows = collect(scratch_dir)
    log.info("Collected %d children with generated PDB + terminal reward", len(rows))
    if len(rows) < 5:
        log.error("Not enough data for correlation (need >=5, got %d)", len(rows))
        return

    reward = np.array([r["terminal_reward"] for r in rows])
    results = {"n_total": len(rows), "overall": {}, "by_timestep": {}}

    for metric in ("rg", "nc_termini"):
        vals = np.array([r[metric] for r in rows])
        results["overall"][metric] = {
            "pearson": _pearson(vals, reward),
            "spearman": _spearman(vals, reward),
        }

    by_t = defaultdict(list)
    for r in rows:
        by_t[r["branch_t"]].append(r)
    for t in sorted(by_t):
        group = by_t[t]
        rew = np.array([g["terminal_reward"] for g in group])
        entry = {"n": len(group)}
        for metric in ("rg", "nc_termini"):
            vals = np.array([g[metric] for g in group])
            entry[metric] = {
                "pearson": _pearson(vals, rew),
                "spearman": _spearman(vals, rew),
            }
        results["by_timestep"][t] = entry

    out_dir = config.EXPERIMENTS_LOG_DIR / "fk_correlation_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 72}")
    print("  FK-Steering correlation test -- cheap geometry vs terminal reward")
    print(f"  n = {len(rows)} generated children across branch timesteps")
    print(f"{'=' * 72}")
    print(f"  OVERALL (Pearson / Spearman correlation with terminal reward):")
    for metric in ("rg", "nc_termini"):
        o = results["overall"][metric]
        print(f"    {metric:<12}  pearson={o['pearson']:+.3f}   spearman={o['spearman']:+.3f}")
    print(f"\n  BY BRANCH TIMESTEP (Spearman; sign shows direction):")
    print(f"    {'t':>5}  {'n':>4}  {'rg':>8}  {'nc_termini':>11}")
    print(f"    {'-' * 32}")
    for t in sorted(by_t):
        e = results["by_timestep"][t]
        print(f"    {t:>5}  {e['n']:>4}  {e['rg']['spearman']:>+8.3f}  "
              f"{e['nc_termini']['spearman']:>+11.3f}")
    print(f"{'=' * 72}")
    print("  (Correlation reported, no recommendation -- see NEXT_STEPS.md §7.1.)")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
