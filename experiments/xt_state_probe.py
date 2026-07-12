"""x_T state-representation probe -- the §3.3 go/no-go gate (NEXT_STEPS.md §3.3).

§3.3 asks whether building the RL policy's state from Genie3's own latents (starting with x_T,
the branch-point noised structure) instead of oracle scalars would help. That is a multi-week
live-GPU build (new per-step latent capture + a one-time latent-augmented dataset + a structural
policy encoder). Before investing, this probe does the cheap directional check -- exactly as the
§7.1 FK-steering correlation test gated the FK apparatus: does the cheapest available latent (x_T)
carry any signal about the terminal reward at all?

Data: the kept live-oracle scratch dirs (RLKNOBS_KEEP_LIVE_SCRATCH=1), the same source
experiments/fk_correlation_test.py reads. Each run branched once and stored `x_T` ([N_res, 3]) in
metadata.json plus per-child terminal metrics. x_T is shared across a run's children (one branch
point), so the unit is the run: ~25 (x_T, best-child terminal reward) pairs.

IMPORTANT CAVEAT (reported, not hidden): n ~= 25 is a WEAK, directional gate, not a definitive
test. A *null* result is not conclusive against §3.3 -- a learned encoder might extract signal
these crude geometric features miss. A clearly *positive* result is a strong green light. The
genuinely decisive gate is this probe combined with the Phase-1 contextual-bandit result (does
scalar-state PPO already match a per-target bandit?), and ultimately the live build itself.

Usage:
    conda run -n genie3 python -m experiments.xt_state_probe
    RLKNOBS_LIVE_SCRATCH=/path/to/scratch conda run -n genie3 python -m experiments.xt_state_probe
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

import config
from oracle.reward_oracle import compute_reward
from experiments.fk_correlation_test import (
    _BRANCH_RE, _pearson, _radius_of_gyration, _spearman,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# The x_T scalar features we probe. All are cheap, coordinate-only summaries of the branch-point
# structure -- a "does anything about the latent geometry predict outcome" first cut.
_FEATURES = ("rg", "coord_spread", "mean_pairwise_dist", "bbox_diag")


def _xt_features(xt: np.ndarray) -> dict[str, float]:
    """Cheap coordinate-only summaries of the x_T [N_res, 3] cloud."""
    center = xt.mean(axis=0)
    dists_to_center = np.linalg.norm(xt - center, axis=1)
    # mean pairwise distance (direct; N is a few hundred so N^2 is fine for a one-off probe)
    diff = xt[:, None, :] - xt[None, :, :]
    pdist = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(len(xt), k=1)
    return {
        "rg": _radius_of_gyration(xt),
        "coord_spread": float(dists_to_center.std()),
        "mean_pairwise_dist": float(pdist[iu].mean()),
        "bbox_diag": float(np.linalg.norm(xt.max(axis=0) - xt.min(axis=0))),
    }


def _best_child_terminal_reward(branch_dir: Path) -> float | None:
    """compute_reward of the best child (by iptm), matching _aggregate_children's selection."""
    best_reward, best_iptm = None, -1.0
    for cm_path in branch_dir.glob("child_*_metrics.json"):
        try:
            m = json.loads(cm_path.read_text())
        except Exception:
            continue
        if m.get("iptm") is None or "error" in m:
            continue
        if m["iptm"] > best_iptm:
            best_iptm = m["iptm"]
            best_reward = compute_reward(m)
    return best_reward


def collect(scratch_dir: Path) -> list[dict]:
    """Per run with a captured x_T: (branch_t, x_T features, best-child terminal reward)."""
    rows = []
    for meta_path in scratch_dir.glob("run_*/branch_t_*/*/metadata.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get("x_T") is None:
            continue
        m = _BRANCH_RE.search(str(meta_path))
        if not m:
            continue
        branch_t = int(m.group(1))
        xt = np.asarray(meta["x_T"], dtype=np.float64)
        if xt.ndim != 2 or xt.shape[0] < 2:
            continue
        reward = _best_child_terminal_reward(meta_path.parent)
        if reward is None:
            continue
        row = {"branch_t": branch_t, "terminal_reward": reward}
        row.update(_xt_features(xt))
        rows.append(row)
    return rows


def main():
    scratch_dir = Path(
        os.environ.get("RLKNOBS_LIVE_SCRATCH", "/pscratch/sd/s/shreyash/rlknobs_live")
    )
    log.info("Collecting x_T + terminal reward from %s ...", scratch_dir)
    rows = collect(scratch_dir)
    log.info("Collected %d runs with x_T and a terminal reward", len(rows))
    if len(rows) < 5:
        log.error("Not enough data for the probe (need >=5, got %d)", len(rows))
        return

    reward = np.array([r["terminal_reward"] for r in rows])
    results = {"n_total": len(rows), "overall": {}, "by_timestep": {}}
    for feat in _FEATURES:
        vals = np.array([r[feat] for r in rows])
        results["overall"][feat] = {"pearson": _pearson(vals, reward),
                                    "spearman": _spearman(vals, reward)}

    by_t = defaultdict(list)
    for r in rows:
        by_t[r["branch_t"]].append(r)
    for t in sorted(by_t):
        grp = by_t[t]
        rew = np.array([g["terminal_reward"] for g in grp])
        entry = {"n": len(grp)}
        for feat in _FEATURES:
            vals = np.array([g[feat] for g in grp])
            entry[feat] = {"pearson": _pearson(vals, rew), "spearman": _spearman(vals, rew)}
        results["by_timestep"][t] = entry

    out_dir = config.EXPERIMENTS_LOG_DIR / "xt_state_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 76}")
    print("  x_T state probe -- branch-point latent geometry vs terminal reward (§3.3 gate)")
    print(f"  n = {len(rows)} runs  (WEAK/DIRECTIONAL -- see docstring caveat)")
    print(f"{'=' * 76}")
    print("  OVERALL (Pearson / Spearman correlation with terminal reward):")
    for feat in _FEATURES:
        o = results["overall"][feat]
        print(f"    {feat:<20}  pearson={o['pearson']:+.3f}   spearman={o['spearman']:+.3f}")
    print(f"\n  BY BRANCH TIMESTEP (Spearman; sign shows direction):")
    header = "    " + f"{'t':>5}  {'n':>4}  " + "  ".join(f"{f:>18}" for f in _FEATURES)
    print(header)
    print("    " + "-" * (len(header) - 4))
    for t in sorted(by_t):
        e = results["by_timestep"][t]
        cells = "  ".join(f"{e[f]['spearman']:>+18.3f}" for f in _FEATURES)
        print(f"    {t:>5}  {e['n']:>4}  {cells}")
    print(f"{'=' * 76}")
    print("  Gate rule (§2b): strong signal -> proceed to Phase 3 live latent build;")
    print("  flat here AND Phase-1 PPO ~= contextual bandit -> §3.3 likely not worth the live cost.")
    print(f"{'=' * 76}\n")


if __name__ == "__main__":
    main()
