"""Unit tests for compute_reward -- callable with fixture metrics dicts, no genie3 needed.

compute_reward is tiered (NEXT_STEPS.md §2.2): designable + hotspot-gated > designable only
> not designable, with an explicit negative reward in the fail tier. Falls back to the old
flat weighted average (_legacy_weighted_average) only when a dict has no success-relevant
signal at all.
"""

from __future__ import annotations

from oracle.reward_oracle import RewardWeights, compute_reward, _legacy_weighted_average


def test_top_tier_requires_designable_and_hotspot_gate():
    metrics = {"iptm": 0.88, "avg_interface_pae": 4.7, "complex_success": True,
               "hotspot_coverage": 0.83}
    r = compute_reward(metrics)
    w = RewardWeights()
    assert r == w.tier_success_base + 0.83 * w.tier_hotspot_scale + 0.88 * w.tier_nuance_scale
    assert r > w.tier_success_base  # comfortably above the tier floor


def test_designable_below_hotspot_threshold_gets_partial_tier_not_top_tier():
    below = {"complex_success": True, "hotspot_coverage": 0.1, "iptm": 0.95}
    above = {"complex_success": True, "hotspot_coverage": 0.9, "iptm": 0.95}
    assert compute_reward(below) < compute_reward(above)
    w = RewardWeights()
    assert compute_reward(below) < w.tier_success_base  # never reaches tier 1's floor


def test_success_increases_reward():
    base = {"iptm": 0.85, "avg_interface_pae": 6.0}
    r_succ = compute_reward({**base, "complex_success": True})
    r_fail = compute_reward({**base, "complex_success": False})
    assert r_succ > r_fail


def test_not_designable_is_negative_and_scales_with_scrmsd():
    close = compute_reward({"complex_success": False, "complex_scrmsd": 2.0})
    far = compute_reward({"complex_success": False, "complex_scrmsd": 20.0})
    assert close < 0.0 and far < 0.0
    assert far < close  # worse structure -> more negative


def test_not_designable_without_scrmsd_uses_fixed_worst_case_penalty():
    w = RewardWeights()
    assert compute_reward({"complex_success": False}) == w.tier_fail_no_scrmsd_penalty


def test_unknown_hotspot_coverage_falls_back_to_partial_tier_default_gap():
    w = RewardWeights()
    r = compute_reward({"complex_success": True})  # no hotspot_coverage key at all
    assert r == w.tier_partial_base - w.tier_partial_unknown_gap / w.tier_partial_scale


def test_missing_success_signal_falls_back_to_legacy_weighted_average():
    # diversity is the one term with no other coupling -> legacy path, reward equals it
    assert abs(compute_reward({"diversity": 0.7}) - 0.7) < 1e-9
    assert compute_reward({}) == 0.0


def test_legacy_fallback_matches_direct_call():
    metrics = {"diversity": 0.4}
    assert compute_reward(metrics) == _legacy_weighted_average(metrics, None, RewardWeights())


def test_hotspot_coverage_threshold_is_swappable():
    metrics = {"complex_success": True, "hotspot_coverage": 0.6, "iptm": 0.9}
    lenient = RewardWeights(hotspot_coverage_threshold=0.5)
    strict = RewardWeights(hotspot_coverage_threshold=0.9)
    assert compute_reward(metrics, weights=lenient) > compute_reward(metrics, weights=strict)
