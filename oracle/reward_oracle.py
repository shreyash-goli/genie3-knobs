"""Reward oracle: the scalar terminal reward + the (offline / live) metric backends.

Two clearly separated pieces, per the spec's "swappable ``compute_reward``" requirement:

* ``compute_reward(metrics, history, weights)`` -- a *pure* function mapping a metrics dict
  to a scalar.  No I/O, no genie3, no dataset.  Unit-testable with fixture dicts (see
  tests/test_reward_oracle.py).  Tune weights here without touching env logic.

* ``RewardOracle`` -- produces the metrics dict for a chosen (target, action).  Two backends:
    - ``OfflineRewardModel``: samples a real logged child from the offline dataset for the
      requested lever cell, with documented back-off when that exact cell is unpopulated.
      This is the offline-first backend used for the whole MVP.
    - ``LiveRewardModel``: STUB -- the extension point that will actually run
      Genie3 -> ProteinMPNN -> ColabFold.  Not implemented (offline-first, by decision).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import config
from instrumentation.trajectory_logger import load_records

# --------------------------------------------------------------------------------------
# Pure reward function
# --------------------------------------------------------------------------------------
@dataclass
class RewardWeights:
    """Weights for the terminal reward.

    ``compute_reward`` is tiered (NEXT_STEPS.md §2.2), not a flat weighted average: crossing
    the designable+hotspot-coverage gate determines which regime a sample is in, and the
    per-regime fields below set the shape *within* that regime. The old flat-average fields
    (``interface_iptm`` etc.) are kept only for ``_legacy_weighted_average``, the fallback
    used when a metrics dict carries no success-relevant signal at all (e.g. partial fixture
    dicts in tests) -- see that function's docstring. Edit freely -- nothing else depends on
    the specific values."""
    success: float = 1.0          # pass/fail (complex_success or thresholded iptm/ipae)
    interface_iptm: float = 0.5   # ipTM, higher better
    interface_ipae: float = 0.5   # interface pAE, lower better
    hotspot_coverage: float = 0.5 # fraction of hotspots contacted (esp. missed ones)
    diversity: float = 0.1        # bonus for being different from siblings/history

    # thresholds used when complex_success is absent
    iptm_pass: float = 0.80
    ipae_pass: float = 10.0
    # normalisation constant for the ipae term (pAE in Angstrom; 30 ~ very bad)
    ipae_scale: float = 30.0

    # -- tiered reward (§2.2) -----------------------------------------------------------
    # Gate: a designable sample only reaches the top tier if it also covers this fraction
    # of the target's hotspot residues. Not specified numerically in the design doc; 0.5
    # ("covers a majority of hotspots") is a reasonable default and is the one knob most
    # worth sweeping once real hotspot_coverage data exists.
    hotspot_coverage_threshold: float = 0.5
    # Tier 1 (designable + hotspot-gated): reward = success_base + coverage*hotspot_scale
    #         + nuance*nuance_scale. ``nuance`` is meant to be iptm_energy (§2.3) -- a
    #         free-energy quantity from pre-softmax PAE logits ColabFold doesn't expose
    #         through this pipeline yet -- so normalized iptm is used as the stand-in until
    #         that plumbing exists.
    tier_success_base: float = 1.0
    tier_hotspot_scale: float = 3.0
    tier_nuance_scale: float = 1.0
    # Tier 2 (designable, coverage below threshold or unknown): partial credit that still
    # beats the fail tier. When hotspot_coverage is None (no PDB/PAE available for this
    # sample) the coverage gap defaults to 0.5 (treat as half-covered) rather than 0 or 1,
    # so an unmeasured sample doesn't get free full credit or unfairly maximal penalty.
    tier_partial_base: float = 0.5
    tier_partial_scale: float = 5.0
    tier_partial_unknown_gap: float = 0.5
    # Tier 3 (not designable): reward = -complex_scrmsd / tier_fail_scale. When scRMSD
    # itself is unavailable (e.g. every ColabFold child failed -- see
    # `_aggregate_children`'s early-return dict), fall back to this fixed penalty, chosen to
    # be worse than a typical scored failure (scRMSD up to ~30 -> -3.0 already, so a total
    # pipeline failure should not look better than a merely-bad structure).
    tier_fail_scale: float = 10.0
    tier_fail_no_scrmsd_penalty: float = -3.0


def _success_term(metrics: dict[str, Any], w: RewardWeights) -> Optional[float]:
    if metrics.get("complex_success") is not None:
        return 1.0 if metrics["complex_success"] else 0.0
    iptm = metrics.get("iptm")
    ipae = metrics.get("avg_interface_pae")
    if iptm is None and ipae is None:
        return None
    ok = True
    if iptm is not None:
        ok = ok and iptm >= w.iptm_pass
    if ipae is not None:
        ok = ok and ipae <= w.ipae_pass
    return 1.0 if ok else 0.0


def compute_reward(
    metrics: dict[str, Any],
    history: Optional[list[dict[str, Any]]] = None,
    weights: Optional[RewardWeights] = None,
) -> float:
    """Scalar terminal reward from an oracle metrics dict (NEXT_STEPS.md §2.2, tiered).

    Three tiers, not a weighted blend:
      1. designable AND hotspot_coverage >= threshold -> success_base + coverage*scale +
         nuance*scale  (unbounded above ~[1, 5]).
      2. designable only -> partial credit, still positive but capped below tier 1.
      3. not designable -> an explicit *negative* reward scaled by scRMSD, so failure is
         unambiguous rather than just "a low positive score".

    ``designable`` reuses the existing complex_success / thresholded-iptm+ipae check
    (``_success_term``). This deliberately makes designability a gate rather than a
    continuous term: unlike the old flat average, ipTM/pAE no longer shape the reward
    smoothly once a sample is designable -- only the tier and (for tier 1) the nuance term
    do. This is the tradeoff the tiered design makes for an unambiguous success signal.

    Falls back to ``_legacy_weighted_average`` when ``metrics`` carries no success-relevant
    signal at all (``_success_term`` returns None, i.e. no complex_success, iptm, or ipae) --
    this keeps ``compute_reward`` well-defined for partial dicts (unit-test fixtures, a
    diversity-only history entry) instead of forcing every caller to populate a full record.

    ``history`` is accepted for signature compatibility with the legacy path and callers
    that pass it; the tiered formula itself does not use it (§2.2 has no diversity term).
    """
    w = weights or RewardWeights()

    designable = _success_term(metrics, w)
    if designable is None:
        return _legacy_weighted_average(metrics, history, w)

    if designable == 1.0:
        cov = metrics.get("hotspot_coverage")
        if cov is not None and cov >= w.hotspot_coverage_threshold:
            iptm = metrics.get("iptm")
            nuance = float(max(0.0, min(1.0, iptm))) if iptm is not None else 0.0
            return w.tier_success_base + cov * w.tier_hotspot_scale + nuance * w.tier_nuance_scale
        gap = (1.0 - cov) if cov is not None else w.tier_partial_unknown_gap
        return w.tier_partial_base - gap / w.tier_partial_scale

    scrmsd = metrics.get("complex_scrmsd")
    if scrmsd is not None:
        return -float(scrmsd) / w.tier_fail_scale
    return w.tier_fail_no_scrmsd_penalty


def _legacy_weighted_average(
    metrics: dict[str, Any],
    history: Optional[list[dict[str, Any]]],
    w: RewardWeights,
) -> float:
    """Pre-§2.2 flat weighted average over whatever terms are present, renormalised over
    the terms that *were* available. Fallback path only -- see ``compute_reward``."""
    history = history or []

    terms: list[tuple[float, float]] = []  # (weight, value in [0,1])

    s = _success_term(metrics, w)
    if s is not None:
        terms.append((w.success, s))

    iptm = metrics.get("iptm")
    if iptm is not None:
        terms.append((w.interface_iptm, float(max(0.0, min(1.0, iptm)))))

    ipae = metrics.get("avg_interface_pae")
    if ipae is not None:
        terms.append((w.interface_ipae, float(max(0.0, 1.0 - ipae / w.ipae_scale))))

    cov = metrics.get("hotspot_coverage")
    if cov is not None:
        terms.append((w.hotspot_coverage, float(max(0.0, min(1.0, cov)))))

    div = metrics.get("diversity")
    if div is not None:
        terms.append((w.diversity, float(max(0.0, min(1.0, div)))))

    if not terms:
        return 0.0
    total_w = sum(wt for wt, _ in terms)
    if total_w == 0:
        return 0.0
    return sum(wt * val for wt, val in terms) / total_w


# --------------------------------------------------------------------------------------
# Offline metric backend (the simulator the env steps against)
# --------------------------------------------------------------------------------------
def _cell_key(target: str, ts: int, mode: str, length_delta: int) -> tuple:
    return (target, ts, mode, length_delta)


class OfflineRewardModel:
    """Samples a real logged child for a requested (target, timestep, hotspot_mode,
    length_delta) lever cell.

    Because the logged sweeps do not populate the full 3x3x3 x timesteps grid (the hotspot
    / length levers were only varied at one timestep), an exact cell may be empty.  We then
    back off along the levers in a *documented, transparent* order and record which back-off
    level was used in ``info['backoff']`` so experiments can report it:

        0  exact cell
        1  drop length_delta match
        2  also drop hotspot_mode match  (timestep-exact, base conditioning)
        3  nearest timestep (same target), any mode/length
        4  target-global (any cell for this target)

    This keeps the env fully defined everywhere while staying grounded in real data.
    """

    BACKOFF_LABELS = {
        0: "exact",
        1: "drop_length",
        2: "drop_hotspot",
        3: "nearest_timestep",
        4: "target_global",
    }

    def __init__(self, records: Optional[list[dict[str, Any]]] = None,
                 source_sweep: Optional[str] = None, rng: Optional[random.Random] = None):
        recs = records if records is not None else load_records()
        if source_sweep is not None:
            recs = [r for r in recs if r.get("source_sweep") == source_sweep]
        self.records = recs
        self.rng = rng or random.Random()
        self._index_cells()

    def _index_cells(self) -> None:
        self.by_cell: dict[tuple, list[dict]] = defaultdict(list)
        self.by_target_ts: dict[tuple, list[dict]] = defaultdict(list)
        self.by_target: dict[str, list[dict]] = defaultdict(list)
        self.timesteps_by_target: dict[str, set] = defaultdict(set)
        for r in self.records:
            key = _cell_key(r["target"], r["branch_timestep"], r["hotspot_mode"],
                            r["length_delta"])
            self.by_cell[key].append(r)
            self.by_target_ts[(r["target"], r["branch_timestep"])].append(r)
            self.by_target[r["target"]].append(r)
            self.timesteps_by_target[r["target"]].add(r["branch_timestep"])

    # -- introspection used by the env / baselines to build a valid action space --------
    def targets(self) -> list[str]:
        return sorted(self.by_target.keys())

    def available_timesteps(self, target: str) -> list[int]:
        return sorted(self.timesteps_by_target.get(target, set()))

    def available_modes(self, target: str) -> list[str]:
        return sorted({r["hotspot_mode"] for r in self.by_target.get(target, [])})

    def available_length_deltas(self, target: str) -> list[int]:
        return sorted({r["length_delta"] for r in self.by_target.get(target, [])})

    # -- sampling -----------------------------------------------------------------------
    def sample(self, target: str, timestep: int, hotspot_mode: str = "all",
               length_delta: int = 0) -> tuple[dict[str, Any], int]:
        """Return (metrics_dict, backoff_level) for the requested lever cell."""
        # 0 exact
        pool = self.by_cell.get(_cell_key(target, timestep, hotspot_mode, length_delta))
        if pool:
            return self.rng.choice(pool), 0
        # 1 drop length
        pool = [r for r in self.by_target_ts.get((target, timestep), [])
                if r["hotspot_mode"] == hotspot_mode]
        if pool:
            return self.rng.choice(pool), 1
        # 2 drop hotspot (timestep-exact)
        pool = self.by_target_ts.get((target, timestep), [])
        if pool:
            return self.rng.choice(pool), 2
        # 3 nearest timestep
        avail = self.available_timesteps(target)
        if avail:
            nearest = min(avail, key=lambda t: abs(t - timestep))
            pool = self.by_target_ts.get((target, nearest), [])
            if pool:
                return self.rng.choice(pool), 3
        # 4 target-global
        pool = self.by_target.get(target, [])
        if pool:
            return self.rng.choice(pool), 4
        raise KeyError(f"No logged data for target {target!r}")

    def cell_stats(self, target: str, timestep: int, hotspot_mode: str = "all",
                   length_delta: int = 0) -> dict[str, Any]:
        """Mean reward / success / n for an exact cell (used by the bandit & fixed
        heuristic to read the ground truth they are competing against)."""
        pool = self.by_cell.get(_cell_key(target, timestep, hotspot_mode, length_delta), [])
        return _summarise_pool(pool)


def _summarise_pool(pool: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(pool)
    if n == 0:
        return {"n": 0, "mean_reward": float("nan"), "success_rate": float("nan")}
    rewards = [compute_reward(r) for r in pool]
    succ = [1.0 if r.get("complex_success") else 0.0 for r in pool]
    return {
        "n": n,
        "mean_reward": sum(rewards) / n,
        "success_rate": sum(succ) / n,
    }


# --------------------------------------------------------------------------------------
# Live backend -- STUB (offline-first; this is the documented extension point)
# --------------------------------------------------------------------------------------
class LiveRewardModel:
    """Extension point for live generation: Genie3 -> ProteinMPNN -> ColabFold.

    Deliberately NOT implemented (the MVP is offline-first by decision).  When wired, this
    should *shell out* to the genie3 conda env (subprocess / SLURM) rather than importing
    genie3 in-process, so the RL stack (SB3, torch>=2.x) and genie3 (torch==2.7.1) stay in
    separate environments.  Keep the ``sample`` signature identical to OfflineRewardModel
    so the env is agnostic to the backend.
    """

    def __init__(self, *args, **kwargs):
        pass

    def sample(self, target: str, timestep: int, hotspot_mode: str = "all",
               length_delta: int = 0):
        raise NotImplementedError(
            "LiveRewardModel is a Stage-4+ stub. Offline-first MVP uses OfflineRewardModel. "
            "Implement by submitting a genie3 generation + eval job and parsing "
            "child_*_metrics.json (same schema as the offline dataset)."
        )
