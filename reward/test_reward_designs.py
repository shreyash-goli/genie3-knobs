"""Unit tests for the candidate reward designs (reward/reward_designs.py).

Run:  conda activate genie3 && python -m pytest reward/test_reward_designs.py -q
"""

from __future__ import annotations

from reward.reward_designs import (
    REWARD_DESIGNS,
    GatedReward,
    WithDevelopability,
    _designable,
    _geometry_term,
    current_plus_coverage_reward,
    current_reward,
)

GOOD = {"complex_success": True, "iptm": 0.85, "avg_interface_pae": 5.0}
BAD = {"complex_success": False, "iptm": 0.30, "avg_interface_pae": 20.0}


# -- _designable -----------------------------------------------------------------------
def test_designable_prefers_complex_success():
    assert _designable({"complex_success": True, "iptm": 0.1}) is True
    assert _designable({"complex_success": False, "iptm": 0.99}) is False


def test_designable_falls_back_to_thresholds():
    assert _designable({"iptm": 0.85, "avg_interface_pae": 5.0}) is True
    assert _designable({"iptm": 0.85, "avg_interface_pae": 15.0}) is False  # pae too high
    assert _designable({"iptm": 0.5, "avg_interface_pae": 5.0}) is False     # iptm too low
    assert _designable({}) is False


# -- current (blind) vs current+coverage -----------------------------------------------
def test_current_is_blind_to_coverage():
    """The production baseline must ignore hotspot_coverage even when it is present."""
    without = current_reward(dict(GOOD))
    with_cov = current_reward({**GOOD, "hotspot_coverage": 1.0})
    with_low = current_reward({**GOOD, "hotspot_coverage": 0.0})
    assert without == with_cov == with_low


def test_current_plus_coverage_uses_coverage():
    """The minimal reform must respond to coverage (its existing 0.5-weighted term fires)."""
    high = current_plus_coverage_reward({**GOOD, "hotspot_coverage": 1.0})
    low = current_plus_coverage_reward({**GOOD, "hotspot_coverage": 0.0})
    assert high > low


# -- geometry term ---------------------------------------------------------------------
def test_geometry_term_prefers_ics_over_pae():
    m_pae = {"avg_interface_pae": 6.0}                 # -> 1 - 6/30 = 0.8
    assert abs(_geometry_term(m_pae) - 0.8) < 1e-6
    m_ics = {"avg_interface_pae": 6.0, "ics": 0.2}     # iCS present -> used instead
    assert abs(_geometry_term(m_ics) - 0.2) < 1e-6


# -- gated (signed, efficiency-scaled, coverage-gated) --------------------------------
def test_gated_failure_is_negative():
    """Property (1): task failure is an explicit NEGATIVE, not a low positive."""
    gated = GatedReward()
    failure = gated({**BAD, "hotspot_coverage": 0.0})
    assert failure < 0.0
    # hybrid floor: every failure is at least as negative as the floor (clearly bad)
    assert failure <= -gated.failure_floor + 1e-9
    # and worse designs (lower quality) are a bit more negative (the slope)
    worse = gated({"complex_success": False, "iptm": 0.05, "avg_interface_pae": 28.0})
    better = gated({"complex_success": False, "iptm": 0.6, "avg_interface_pae": 12.0})
    assert worse < better < 0.0
    # floor+slope covers all failures in [-(floor+slope), -floor]
    assert -(gated.failure_floor + gated.failure_slope) - 1e-9 <= worse
    assert better <= -gated.failure_floor + 1e-9


def test_gated_failure_mode_is_configurable():
    """The floor/slope split recovers flat and graded as special cases."""
    m = {"complex_success": False, "iptm": 0.6, "avg_interface_pae": 12.0}  # quality ~0.6
    flat = GatedReward(failure_floor=1.0, failure_slope=0.0)
    graded = GatedReward(failure_floor=0.0, failure_slope=1.0)
    assert flat(m) == -1.0                       # flat: every failure is exactly -1
    assert abs(graded(m) - -(1.0 - 0.6)) < 1e-9  # graded: -(1 - quality)


def test_gated_success_scales_with_efficiency():
    """Property (2): among successes, higher interface efficiency earns a larger positive."""
    gated = GatedReward()
    hi = gated({"complex_success": True, "iptm": 0.95, "avg_interface_pae": 3.0,
                "hotspot_coverage": 1.0})
    lo = gated({"complex_success": True, "iptm": 0.81, "avg_interface_pae": 9.5,
                "hotspot_coverage": 1.0})
    assert hi > lo > 0.0


def test_gated_impostor_sits_between_failure_and_success():
    """Coverage gate: a designable but epitope-missing impostor collapses toward 0 -- above
    real failure (<0), below genuine on-target success."""
    gated = GatedReward()
    genuine = gated({**GOOD, "hotspot_coverage": 1.0})
    impostor = gated({**GOOD, "hotspot_coverage": 0.0})
    failure = gated({**BAD, "hotspot_coverage": 0.0})
    assert failure < 0.0 <= impostor < genuine
    assert impostor == 0.0


def test_gated_missing_coverage_policy():
    m = dict(GOOD)  # designable, no coverage key
    assert GatedReward(coverage_missing="neutral")(m) > 0.0   # gate disabled -> ungated success
    assert GatedReward(coverage_missing="strict")(m) == 0.0   # missing coverage -> gated out


def test_developability_wrapper_penalises_bad_sequence():
    base = GatedReward()
    penalised = WithDevelopability(base)
    good = {**GOOD, "hotspot_coverage": 1.0, "gravy": -0.5, "net_charge": 3.0}
    bad = {**GOOD, "hotspot_coverage": 1.0, "gravy": 1.5, "net_charge": 20.0}
    assert penalised(good) == base(good)              # in-band: no penalty
    assert penalised(bad) < base(bad)                 # out-of-band: strictly lower
    assert base(bad) - penalised(bad) <= 0.30 + 1e-9  # bounded by max_total
    assert penalised.design_name == "gated+dev"


# -- registry --------------------------------------------------------------------------
def test_registry_designs_all_callable_and_finite():
    for name, fn in REWARD_DESIGNS.items():
        val = fn({**GOOD, "hotspot_coverage": 0.8})
        assert isinstance(val, float)
        assert val == val  # not NaN
