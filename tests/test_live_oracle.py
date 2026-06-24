"""Tests for live_oracle helpers: x_T capture, developability filter, _aggregate_children."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from oracle.live_oracle import (
    _aggregate_children,
    _gravy,
    _net_charge_ph7,
    _passes_developability,
    _parse_child_metrics,
)


# ---------------------------------------------------------------------------
# GRAVY / charge helpers
# ---------------------------------------------------------------------------

class TestGravy:
    def test_poly_ile_is_hydrophobic(self):
        assert _gravy("IIIIII") > 0

    def test_poly_glu_is_hydrophilic(self):
        assert _gravy("EEEEEE") < 0

    def test_empty_seq_is_zero(self):
        assert _gravy("") == 0.0

    def test_unknown_aas_ignored(self):
        # 'X' is not in the KD table; should not raise
        val = _gravy("IIXII")
        assert val == pytest.approx(_gravy("IIII"), rel=1e-3)


class TestNetCharge:
    def test_poly_lys_positive(self):
        assert _net_charge_ph7("KKKK") > 0

    def test_poly_glu_negative(self):
        assert _net_charge_ph7("EEEE") < 0

    def test_balanced(self):
        assert _net_charge_ph7("KE") == pytest.approx(0.0)


class TestPassesDevelopability:
    def test_hydrophilic_passes(self):
        ok, g, c = _passes_developability("EEEE")
        assert ok
        assert g < 0

    def test_hydrophobic_fails(self):
        ok, g, c = _passes_developability("IIIIIIII")
        assert not ok
        assert g > 0

    def test_extreme_charge_fails(self):
        # 12 K residues → net_charge = 12 > 10
        ok, g, c = _passes_developability("K" * 12)
        assert not ok
        assert abs(c) > 10

    def test_moderate_charge_passes(self):
        ok, g, c = _passes_developability("KKKKK" + "EEEEE")  # net = 0
        assert ok


# ---------------------------------------------------------------------------
# _aggregate_children
# ---------------------------------------------------------------------------

def _make_child(iptm=0.85, success=True, seq="EEEEEEEEEE"):
    return {
        "iptm": iptm,
        "avg_interface_pae": 5.0,
        "complex_success": success,
        "binder_seq": seq,
    }


class TestAggregateChildren:
    def test_empty_children_returns_zeros(self):
        m = _aggregate_children([])
        assert m["iptm"] == 0.0
        assert m["x_T"] is None

    def test_best_child_by_iptm_selected(self):
        children = [_make_child(iptm=0.6), _make_child(iptm=0.9)]
        m = _aggregate_children(children)
        assert m["iptm"] == pytest.approx(0.9)

    def test_developability_filter_removes_hydrophobic(self):
        # one hydrophilic (passes) and one very hydrophobic (fails)
        good = _make_child(iptm=0.5, seq="EEEEEEEEEE")   # GRAVY < 0, passes
        bad = _make_child(iptm=0.95, seq="I" * 20)       # GRAVY >> 0, fails
        m = _aggregate_children([good, bad])
        # bad child should be filtered; best should be the good one
        assert m["iptm"] == pytest.approx(0.5)

    def test_all_fail_filter_falls_back_to_unfiltered(self):
        # all hydrophobic — fallback keeps the pool, best-by-iptm from original
        children = [_make_child(iptm=0.7, seq="I" * 20), _make_child(iptm=0.9, seq="I" * 20)]
        m = _aggregate_children(children)
        assert m["iptm"] == pytest.approx(0.9)  # still picks best

    def test_gravy_and_charge_populated(self):
        m = _aggregate_children([_make_child(seq="EEEEEEEEEE")])
        assert m["gravy"] is not None
        assert m["net_charge"] is not None

    def test_x_T_parsed_from_list(self):
        raw = np.random.randn(50, 3).astype(np.float32)
        m = _aggregate_children([_make_child()], x_T=raw.tolist())
        assert m["x_T"] is not None
        assert m["x_T"].shape == (50, 3)
        np.testing.assert_allclose(m["x_T"], raw, atol=1e-5)

    def test_x_T_none_when_not_provided(self):
        m = _aggregate_children([_make_child()])
        assert m["x_T"] is None

    def test_diversity_from_rmsd(self):
        m = _aggregate_children([_make_child()], mean_pairwise_rmsd=5.0)
        assert m["diversity"] == pytest.approx(0.5)  # 5.0 / 10.0


# ---------------------------------------------------------------------------
# _parse_child_metrics — reads child_*_metrics.json from a temp dir
# ---------------------------------------------------------------------------

class TestParseChildMetrics:
    def test_reads_all_child_files(self, tmp_path):
        for i in range(3):
            (tmp_path / f"child_{i}_metrics.json").write_text(
                json.dumps({"iptm": 0.7 + i * 0.1, "child_id": i})
            )
        results = _parse_child_metrics(tmp_path)
        assert len(results) == 3
        assert all("iptm" in r for r in results)

    def test_skips_malformed_json(self, tmp_path):
        (tmp_path / "child_0_metrics.json").write_text("{bad json}")
        (tmp_path / "child_1_metrics.json").write_text(json.dumps({"iptm": 0.8}))
        results = _parse_child_metrics(tmp_path)
        assert len(results) == 1

    def test_empty_dir_returns_empty(self, tmp_path):
        assert _parse_child_metrics(tmp_path) == []
