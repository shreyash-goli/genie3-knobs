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
    """Weights for the terminal reward.  All terms are scaled to ~[0, 1] before weighting
    so the weights are directly interpretable.  Edit freely -- nothing else depends on the
    specific values."""
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
    """Scalar terminal reward from an oracle metrics dict.

    Robust to missing keys: any term whose inputs are ``None`` is dropped and the reward is
    renormalised over the terms that *were* available, so partial metrics dicts (e.g. in
    unit tests) never raise and never silently count a missing metric as zero.

    ``history`` is the list of metrics dicts already generated this round for the same
    target; used only by the diversity term (penalise near-duplicates).
    """
    w = weights or RewardWeights()
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
