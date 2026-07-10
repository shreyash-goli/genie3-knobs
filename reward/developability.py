"""Sequence-only developability terms for binder design (NEXT_STEPS.md §"Biophysical").

Every term here is computed from ``binder_seq`` alone -- no oracle, no GPU, no structure.
That is the whole point: these can be applied retroactively to every logged record (all 4075
have a binder_seq) and to any future design at zero marginal cost, independent of GPU budget.

Tier-1 panel (the "compute today, no external model" row of the §Biophysical table):

  term              role            healthy range      source
  ----------------  --------------  -----------------  -------------------------------
  GRAVY             hard filter     <= 0               Kyte-Doolittle (live_oracle._gravy)
  net charge @ pH7  hard filter     |q| <= 10          live_oracle._net_charge_ph7
  isoelectric pt    soft penalty    pI in [6, 9]       BioPython IsoelectricPoint
  instability idx   soft penalty    < 40               BioPython ProtParam

"Hard" terms gate a candidate out before it wastes downstream effort (already enforced in
``live_oracle._aggregate_children``); "soft" terms subtract a small bounded penalty from the
structural reward so a formulation-marginal design is nudged, not eliminated. Which role each
term plays is a decision, not a fact -- exposed via ``DevelopabilityWeights`` so it can change
without touching reward code.

BioPython is used when present (it is, in the genie3 env); pI/instability degrade to None if
it is ever missing, so importing this module never hard-fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from oracle.live_oracle import _gravy, _net_charge_ph7

try:  # BioPython is present in the genie3 env; stay importable if it ever isn't.
    from Bio.SeqUtils.ProtParam import ProteinAnalysis  # type: ignore
    _HAVE_BIOPYTHON = True
except Exception:  # noqa: BLE001
    _HAVE_BIOPYTHON = False

# 20 canonical amino acids; ProtParam raises on anything else (X, gaps, lowercase junk).
_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _clean(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in _AA)


def isoelectric_point(seq: str) -> Optional[float]:
    """Predicted isoelectric point. None if BioPython is unavailable or the sequence is empty
    after cleaning. pI near physiological pH (7.4) sits in a low-solubility window."""
    seq = _clean(seq)
    if not seq or not _HAVE_BIOPYTHON:
        return None
    try:
        return float(ProteinAnalysis(seq).isoelectric_point())
    except Exception:  # noqa: BLE001
        return None


def instability_index(seq: str) -> Optional[float]:
    """Guruprasad instability index. > 40 predicts an unstable (in-vivo-degradation-prone)
    protein. None if BioPython is unavailable or the sequence is empty after cleaning."""
    seq = _clean(seq)
    if not seq or not _HAVE_BIOPYTHON:
        return None
    try:
        return float(ProteinAnalysis(seq).instability_index())
    except Exception:  # noqa: BLE001
        return None


def developability_panel(seq: str) -> dict[str, Any]:
    """The full Tier-1 panel for one binder sequence. Missing (None) values propagate rather
    than defaulting, so a downstream penalty can drop a term it can't compute."""
    return {
        "gravy": _gravy(seq) if seq else None,
        "net_charge": _net_charge_ph7(seq) if seq else None,
        "isoelectric_point": isoelectric_point(seq),
        "instability_index": instability_index(seq),
    }


@dataclass
class DevelopabilityWeights:
    """Soft-constraint bands + per-term penalty weights. A term contributes
    ``weight * (fractional distance outside its band)``, clipped so no single term can exceed
    its weight; the total soft penalty is clipped to ``max_total``. All ranges are editable
    without touching reward code -- this is where the constraint/objective split lives."""
    # bands (healthy interval or one-sided threshold)
    gravy_max: float = 0.0
    charge_abs_max: float = 10.0
    pi_low: float = 6.0
    pi_high: float = 9.0
    instability_max: float = 40.0
    # per-term weights (contribution to the penalty when fully out of band).
    # pI/instability default to 0.0: on this binder distribution 95%/89% of designs fall
    # outside the antibody-calibrated [6,9] / <40 bands (Tier A), so as literal thresholds
    # they are a near-constant offset, not a discriminator. Recalibrate the bands to the
    # binder distribution before re-enabling; GRAVY/charge (21% out-of-band) stay active.
    w_gravy: float = 0.10
    w_charge: float = 0.10
    w_pi: float = 0.0
    w_instability: float = 0.0
    # scale factors that turn an absolute excess into a ~[0,1] fraction
    gravy_scale: float = 1.0      # GRAVY excess of 1.0 -> full weight
    charge_scale: float = 10.0    # 10 charge units past the cap -> full weight
    pi_scale: float = 2.0         # 2 pH units outside the band -> full weight
    instability_scale: float = 40.0  # 40 index units past 40 -> full weight
    max_total: float = 0.30


def _band_excess_high(value: float, cap: float, scale: float) -> float:
    return max(0.0, (value - cap) / scale)


def _band_excess_interval(value: float, low: float, high: float, scale: float) -> float:
    if value < low:
        return (low - value) / scale
    if value > high:
        return (value - high) / scale
    return 0.0


def soft_penalty(metrics: dict[str, Any],
                 weights: Optional[DevelopabilityWeights] = None) -> float:
    """Bounded developability penalty in [0, max_total] from whatever sequence terms are
    present in ``metrics`` (``gravy``, ``net_charge``, ``isoelectric_point``,
    ``instability_index``). Terms that are None are skipped. Pure function of the metrics
    dict -- no sequence parsing here, so it stays cheap inside a reward call."""
    w = weights or DevelopabilityWeights()
    pen = 0.0

    gravy = metrics.get("gravy")
    if gravy is not None:
        pen += w.w_gravy * min(1.0, _band_excess_high(float(gravy), w.gravy_max, w.gravy_scale))

    charge = metrics.get("net_charge")
    if charge is not None:
        pen += w.w_charge * min(1.0, _band_excess_high(abs(float(charge)), w.charge_abs_max,
                                                       w.charge_scale))

    pi = metrics.get("isoelectric_point")
    if pi is not None:
        pen += w.w_pi * min(1.0, _band_excess_interval(float(pi), w.pi_low, w.pi_high,
                                                       w.pi_scale))

    inst = metrics.get("instability_index")
    if inst is not None:
        pen += w.w_instability * min(1.0, _band_excess_high(float(inst), w.instability_max,
                                                            w.instability_scale))

    return float(min(w.max_total, pen))


def attach_panel(metrics: dict[str, Any]) -> dict[str, Any]:
    """Populate the four Tier-1 developability terms on a metrics dict from its ``binder_seq``
    (only filling terms that are absent/None). Returns the same dict for chaining."""
    seq = metrics.get("binder_seq")
    if not seq:
        return metrics
    panel = developability_panel(seq)
    for k, v in panel.items():
        if metrics.get(k) is None:
            metrics[k] = v
    return metrics
