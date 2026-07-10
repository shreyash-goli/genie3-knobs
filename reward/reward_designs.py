"""Candidate terminal-reward designs for the reward-function reform (NEXT_STEPS.md §2).

Every design is a pure ``metrics_dict -> float`` callable with the SAME signature as
``oracle.reward_oracle.compute_reward`` so an env can swap one in without any other change.
They are deliberately kept out of ``oracle/`` so evaluating them cannot perturb the live
training path; a winner gets promoted into ``compute_reward`` only after the experiments in
this folder justify it.

Designs:

* ``current``            -- today's flat, renormalised weighted average (the baseline). Blind
                            to contact geometry because ``hotspot_coverage`` is always None in
                            the logged data, so its 0.5-weighted term is silently dropped.
* ``current+coverage``   -- the minimal reform: the same flat form with the already-present
                            0.5-weighted coverage term simply allowed to fire.
* ``gated``              -- the RECOMMENDED design. A gated reward that delivers the two
                            properties a tiered scheme is really after, WITHOUT literal tiers
                            or any unavailable term:
                              (1) task FAILURE is explicitly NEGATIVE (not merely a low
                                  positive), so 'failed' is unambiguously distinguished from
                                  'succeeded but mediocre';
                              (2) on SUCCESS the positive magnitude scales with efficiency
                                  (iptm + interface geometry) and is GATED by hotspot coverage,
                                  so an epitope-missing impostor collapses toward 0.
                            The geometry term prefers contact-restricted iCS over the global
                            interface-pAE that is blind to WHICH residues are contacted.
* ``gated+dev``          -- ``gated`` wrapped with the sequence-only developability penalty.

The literal tiered reward that used to live here was a copy of Proteina-Complexa's beam-search
scoring function -- a different policy for a different problem, and it needed ``iptm_energy``
(raw pre-softmax PAE logits, not persisted anywhere). It was removed; ``gated`` reproduces the
two behaviours that actually mattered (harsh-failure + efficiency-scaled-success) with terms we
have.

Which metrics each design consumes (so you know what data a comparison needs):

    term            current   current+coverage   gated
    iptm              yes           yes            yes
    avg_interface_pae yes           yes            fallback*
    hotspot_coverage  (dropped)     yes            GATE
    ics               no            no             preferred*
    gravy/net_charge  no            no             via gated+dev only

    *  gated uses iCS as its geometry term when present, else falls back to the global
       (1 - pae/scale). iCS is contact-restricted, which is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from oracle.reward_oracle import compute_reward as _current_compute_reward
from reward.developability import DevelopabilityWeights, soft_penalty

RewardFn = Callable[..., float]


# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------
def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _designable(metrics: dict[str, Any], iptm_pass: float = 0.80,
                ipae_pass: float = 10.0) -> bool:
    """The existing pass/fail gate: complex_success if given, else iptm/pae thresholds.
    Matches ``oracle.reward_oracle._success_term`` semantics so the designs agree with the
    baseline on what 'designable' means."""
    cs = metrics.get("complex_success")
    if cs is not None:
        return bool(cs)
    iptm = metrics.get("iptm")
    ipae = metrics.get("avg_interface_pae")
    if iptm is None and ipae is None:
        return False
    ok = True
    if iptm is not None:
        ok = ok and iptm >= iptm_pass
    if ipae is not None:
        ok = ok and ipae <= ipae_pass
    return ok


def _geometry_term(metrics: dict[str, Any], ipae_scale: float = 30.0) -> float:
    """Interface geometry quality in [0, 1]. Prefers contact-restricted iCS (which knows
    WHICH residues are contacted); falls back to (1 - pae/scale) from the global interface
    pAE only when iCS is absent."""
    ics = metrics.get("ics")
    if ics is not None:
        return _clip01(ics)
    ipae = metrics.get("avg_interface_pae")
    if ipae is not None:
        return _clip01(1.0 - ipae / ipae_scale)
    return 0.0


def _resolve_coverage(metrics: dict[str, Any], missing: str) -> Optional[float]:
    """Apply the missing-coverage policy (the §2.1 renormalisation decision, made explicit).

    ``strict``  -> a missing coverage counts as 0.0 (you cannot earn the gate by having no
                   geometry metric; correct once coverage is populated everywhere).
    ``neutral`` -> a missing coverage disables the gate (falls back to ungated behaviour);
                   needed for back-compat scoring of legacy records that predate coverage.
    """
    cov = metrics.get("hotspot_coverage")
    if cov is not None:
        return _clip01(cov)
    if missing == "strict":
        return 0.0
    return None  # neutral: caller treats None as "no gate"


# --------------------------------------------------------------------------------------
# Design 0 -- current production reward (baseline)
# --------------------------------------------------------------------------------------
def current_reward(metrics: dict[str, Any],
                   history: Optional[list[dict[str, Any]]] = None) -> float:
    """The TRUE production baseline: ``compute_reward`` as it behaves on today's data, where
    every logged record has ``hotspot_coverage``/``ics`` == None so the geometry terms are
    always dropped. We force them absent here so this stays the geometry-blind baseline even
    when coverage has been retroactively attached to a row for the reform designs to use."""
    m = dict(metrics)
    m["hotspot_coverage"] = None
    m["ics"] = None
    return _current_compute_reward(m, history)


current_reward.design_name = "current"  # type: ignore[attr-defined]


def current_plus_coverage_reward(metrics: dict[str, Any],
                                 history: Optional[list[dict[str, Any]]] = None) -> float:
    """The *minimal* reform: the existing flat weighted average with nothing restructured --
    just let its already-present 0.5-weighted ``hotspot_coverage`` term fire (it never does
    today only because the data is None). Isolates 'populate the term' from 'restructure the
    reward': compare its coverage-sensitivity to the gated/tiered designs to see how much of
    the reform's effect needs the new *structure* vs. just the new *data*."""
    return _current_compute_reward(metrics, history)


current_plus_coverage_reward.design_name = "current+coverage"  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# Design -- gated (RECOMMENDED): signed, efficiency-scaled, coverage-gated
# --------------------------------------------------------------------------------------
@dataclass
class GatedReward:
    """A gated reward with the two behaviours a tiered scheme is really after -- an explicit
    failure penalty and efficiency-scaled success -- but no literal tiers and no unavailable
    term. One branch on the success/failure boundary; continuous within each branch:

        quality = 0.5*iptm + 0.5*geometry                 (geometry = iCS if present, else 1-pae/scale)
        SUCCESS (designable):  reward = coverage * (success_base + success_eff * quality)   >= 0
        FAILURE (not desig.):  reward = -(failure_floor + failure_slope * (1 - quality))    <  0

    Property (1), harsh failure: the failure branch is a strong flat FLOOR plus a small SLOPE.
    The floor makes every failure clearly bad -- with the defaults all failures land in
    [-(floor+slope), -floor] = [-1.0, -0.7], strictly below any success, so 'failed' is
    unambiguous. The slope keeps a gradient within the failure region (a worse fold is a bit
    more negative), so the value function has something to fit and the policy is nudged toward
    the success boundary rather than being starved by a flat penalty (NEXT_STEPS.md §6). The
    floor/slope split covers all three modes: hybrid (default), flat (slope=0), graded
    (floor=0). Property (2), efficiency-scaled success: the positive magnitude grows with
    ``quality`` and is multiplied by ``coverage`` -- so an epitope-missing impostor
    (designable, coverage->0) collapses to ~0, between real failure (<0) and on-target success.

    Default constants: success in [0, 1.5] (gated), failure in [-1.0, -0.7]. ``coverage_missing``
    selects the §2.1 policy when coverage is absent (neutral -> gate disabled, i.e. the reward
    still works on legacy records with the failure/efficiency behaviour, just ungated).
    Developability is orthogonal -- wrap in ``WithDevelopability`` to add the sequence penalty.
    """
    iptm_pass: float = 0.80
    ipae_pass: float = 10.0
    ipae_scale: float = 30.0
    success_base: float = 0.5      # positive baseline for crossing the success gate
    success_eff: float = 1.0       # how much success magnitude scales with efficiency/quality
    failure_floor: float = 0.7     # flat penalty every failure gets (the "really punish" part)
    failure_slope: float = 0.3     # extra penalty scaled by how bad the failure is (gradient)
    coverage_missing: str = "neutral"  # "neutral" | "strict"
    design_name: str = "gated"

    def __call__(self, metrics: dict[str, Any],
                 history: Optional[list[dict[str, Any]]] = None) -> float:
        quality = 0.5 * _clip01(metrics.get("iptm") or 0.0) + \
            0.5 * _geometry_term(metrics, self.ipae_scale)
        if _designable(metrics, self.iptm_pass, self.ipae_pass):
            gate = _resolve_coverage(metrics, self.coverage_missing)
            g = 1.0 if gate is None else gate
            return float(g * (self.success_base + self.success_eff * quality))
        return float(-(self.failure_floor + self.failure_slope * (1.0 - quality)))


# --------------------------------------------------------------------------------------
# Developability layer -- composable over ANY design
# --------------------------------------------------------------------------------------
@dataclass
class WithDevelopability:
    """Wrap any reward design and subtract the sequence-only developability soft penalty
    (GRAVY / net charge / pI / instability index -- all computed from ``binder_seq``, no GPU).

    Composable so developability is orthogonal to the structural-reward choice: you can add
    it to the flat or gated design without duplicating the penalty logic. The penalty
    is bounded (``DevelopabilityWeights.max_total``, default 0.30) so it nudges rather than
    dominates. Requires the four terms to be present on the metrics dict -- use
    ``developability.attach_panel`` (or the offline analyses in this folder) to populate them
    from ``binder_seq`` first; absent terms are simply skipped."""
    base: RewardFn
    weights: DevelopabilityWeights = field(default_factory=DevelopabilityWeights)
    design_name: str = ""

    def __post_init__(self) -> None:
        if not self.design_name:
            self.design_name = design_name(self.base) + "+dev"

    def __call__(self, metrics: dict[str, Any],
                 history: Optional[list[dict[str, Any]]] = None) -> float:
        return float(self.base(metrics, history) - soft_penalty(metrics, self.weights))


# --------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------
def design_name(fn: RewardFn) -> str:
    return getattr(fn, "design_name", getattr(fn, "__name__", repr(fn)))


#: The designs an experiment iterates over. ``coverage_missing`` is left at the per-design
#: default here; experiments that score legacy records override it explicitly. The ``+dev``
#: entry shows developability composed onto the recommended design; developability is
#: orthogonal, so it can wrap any of the others too.
REWARD_DESIGNS: dict[str, RewardFn] = {
    "current": current_reward,                          # geometry-blind production baseline
    "current+coverage": current_plus_coverage_reward,   # minimal reform: populate existing term
    "gated": GatedReward(),                             # RECOMMENDED: signed, gated, efficiency-scaled
    "gated+dev": WithDevelopability(GatedReward()),     # + sequence developability penalty
}
