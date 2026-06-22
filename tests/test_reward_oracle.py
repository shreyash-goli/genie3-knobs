"""Unit tests for compute_reward -- callable with fixture metrics dicts, no genie3 needed."""

from __future__ import annotations

from oracle.reward_oracle import RewardWeights, compute_reward


def test_compute_reward_returns_float_on_full_dict():
    metrics = {"iptm": 0.88, "avg_interface_pae": 4.7, "complex_success": True,
               "hotspot_coverage": 0.83, "diversity": 0.5}
    r = compute_reward(metrics)
    assert isinstance(r, float)
    assert 0.0 <= r <= 1.0


def test_success_increases_reward():
    base = {"iptm": 0.85, "avg_interface_pae": 6.0}
    r_succ = compute_reward({**base, "complex_success": True})
    r_fail = compute_reward({**base, "complex_success": False})
    assert r_succ > r_fail


def test_lower_ipae_is_better():
    good = compute_reward({"avg_interface_pae": 2.0})
    bad = compute_reward({"avg_interface_pae": 25.0})
    assert good > bad


def test_missing_keys_do_not_raise_and_renormalise():
    # diversity is the one term with no other coupling -> reward equals it (renormalised)
    assert abs(compute_reward({"diversity": 0.7}) - 0.7) < 1e-9
    # a lone iptm also drives the thresholded success term, so it must stay finite in range
    r = compute_reward({"iptm": 0.6})
    assert isinstance(r, float) and 0.0 <= r <= 1.0


def test_empty_metrics_is_zero():
    assert compute_reward({}) == 0.0


def test_weights_are_swappable():
    metrics = {"iptm": 1.0, "avg_interface_pae": 30.0}  # great iptm, terrible ipae
    only_iptm = RewardWeights(success=0.0, interface_iptm=1.0, interface_ipae=0.0)
    only_ipae = RewardWeights(success=0.0, interface_iptm=0.0, interface_ipae=1.0)
    assert compute_reward(metrics, weights=only_iptm) > compute_reward(metrics, weights=only_ipae)
