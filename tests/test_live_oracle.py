"""Tests for live_oracle helpers: x_T capture, developability filter, _aggregate_children."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

import oracle.live_oracle as live_oracle_module
from oracle.live_oracle import (
    LiveRewardModel,
    MultiGPULiveRewardModel,
    NoDatasetVariant,
    _aggregate_children,
    _build_selection,
    _compute_hotspot_coverage,
    _compute_ics,
    _conda_run,
    _gravy,
    _interface_contact_pairs,
    _needs_ablation_config,
    _net_charge_ph7,
    _parse_pdb_cb_coords,
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


# ---------------------------------------------------------------------------
# _conda_run — CUDA_VISIBLE_DEVICES scoping for GPU pinning
# ---------------------------------------------------------------------------

class TestCondaRunExtraEnv:
    def test_no_extra_env_passes_none(self, monkeypatch):
        """Unchanged behavior: without extra_env, subprocess.run gets env=None (inherits
        the parent process's environment untouched)."""
        captured = {}

        def fake_run(*args, **kwargs):
            captured.update(kwargs)
            class _R:
                returncode = 0
                stdout = ""
                stderr = ""
            return _R()

        monkeypatch.setattr(live_oracle_module.subprocess, "run", fake_run)
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "genie3")
        _conda_run(["python", "-c", "pass"], "genie3", cwd=Path("."))
        assert captured["env"] is None

    def test_extra_env_merges_over_current_environment(self, monkeypatch):
        captured = {}

        def fake_run(*args, **kwargs):
            captured.update(kwargs)
            class _R:
                returncode = 0
                stdout = ""
                stderr = ""
            return _R()

        monkeypatch.setattr(live_oracle_module.subprocess, "run", fake_run)
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "genie3")
        monkeypatch.setenv("SOME_UNRELATED_VAR", "keep-me")
        _conda_run(
            ["python", "-c", "pass"], "genie3", cwd=Path("."),
            extra_env={"CUDA_VISIBLE_DEVICES": "2"},
        )
        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2"
        assert captured["env"]["SOME_UNRELATED_VAR"] == "keep-me"


class TestLiveRewardModelDevicePinning:
    """LiveRewardModel._run_branching / _run_eval must scope CUDA_VISIBLE_DEVICES to
    self.device -- this is the only way to pin the branching step (no --device flag
    exists on branching_wrapper.py), and must stay consistent with the eval step so a
    single LiveRewardModel instance uses exactly one physical GPU end-to-end."""

    def test_run_branching_scopes_cuda_visible_devices(self, monkeypatch, tmp_path):
        captured = {}

        def fake_conda_run(cmd, env_name, cwd, timeout=900, extra_env=None):
            captured["cmd"] = cmd
            captured["extra_env"] = extra_env
            return ""

        monkeypatch.setattr(live_oracle_module, "_conda_run", fake_conda_run)
        model = LiveRewardModel(device=3)
        model._run_branching(
            out_dir=tmp_path, timestep=800, selection="01_bhrf1", num_children=5,
            config_yaml=Path("/fake/experiment_trajectory_branching.yaml"),
        )
        assert captured["extra_env"] == {"CUDA_VISIBLE_DEVICES": "3"}
        assert "/fake/experiment_trajectory_branching.yaml" in captured["cmd"]

    def test_run_eval_scopes_cuda_visible_devices_and_uses_device_zero(self, monkeypatch, tmp_path):
        captured = {}

        def fake_conda_run(cmd, env_name, cwd, timeout=900, extra_env=None):
            captured["cmd"] = cmd
            captured["extra_env"] = extra_env
            return ""

        monkeypatch.setattr(live_oracle_module, "_conda_run", fake_conda_run)
        model = LiveRewardModel(device=3)
        model._run_eval(
            sweep_root=tmp_path, problem="01_bhrf1", timestep=800,
            dataset_dir=Path("/fake/binderbench"),
        )
        assert captured["extra_env"] == {"CUDA_VISIBLE_DEVICES": "3"}
        assert "/fake/binderbench" in captured["cmd"]
        # Once CUDA_VISIBLE_DEVICES restricts the subprocess to one device, that device is
        # re-numbered 0 from its point of view -- passing self.device ("3") to eval.py's own
        # --device flag here would ask for a device that doesn't exist in its restricted view.
        device_flag_idx = captured["cmd"].index("--device") + 1
        assert captured["cmd"][device_flag_idx] == "0"


# ---------------------------------------------------------------------------
# MultiGPULiveRewardModel — device isolation + round-robin dispatch
# ---------------------------------------------------------------------------

class _FakeLiveRewardModel:
    """Stands in for LiveRewardModel: no genie3, just records which device ran each
    call and detects if two calls on the *same* device ever overlap."""

    _active: set = set()

    def __init__(self, device: int = 0, **kwargs):
        self.device = device

    def sample(self, target, timestep, hotspot_mode="all", length_delta=0):
        if self.device in _FakeLiveRewardModel._active:
            raise RuntimeError(f"device {self.device} double-booked")
        _FakeLiveRewardModel._active.add(self.device)
        try:
            time.sleep(0.03)
        finally:
            _FakeLiveRewardModel._active.discard(self.device)
        return {"iptm": 0.5, "device_used": self.device}, 0


class TestMultiGPULiveRewardModel:
    def setup_method(self):
        _FakeLiveRewardModel._active = set()

    def test_rejects_empty_devices(self):
        with pytest.raises(ValueError):
            MultiGPULiveRewardModel(devices=[])

    def test_round_robin_device_assignment(self, monkeypatch):
        monkeypatch.setattr(live_oracle_module, "LiveRewardModel", _FakeLiveRewardModel)
        multi = MultiGPULiveRewardModel(devices=[0, 1, 2])
        try:
            futures = [multi.submit(i, "t", 800, "all", 0) for i in range(9)]
            results = [f.result() for f in futures]
        finally:
            multi.shutdown()
        for i, (metrics, backoff) in enumerate(results):
            assert metrics["device_used"] == i % 3

    def test_no_double_booking_across_concurrent_calls(self, monkeypatch):
        """Regression guard: a shared thread pool (instead of one single-worker executor
        per device) would let a freed worker pick up the next queued item regardless of
        which device it targets, risking two calls on the same device running at once."""
        monkeypatch.setattr(live_oracle_module, "LiveRewardModel", _FakeLiveRewardModel)
        multi = MultiGPULiveRewardModel(devices=[0, 1, 2, 3])
        try:
            futures = [multi.submit(i, "t", 800, "all", 0) for i in range(40)]
            # .result() re-raises if sample() detected double-booking
            results = [f.result() for f in futures]
        finally:
            multi.shutdown()
        assert len(results) == 40


# ---------------------------------------------------------------------------
# _build_selection / _needs_ablation_config -- dataset-variant registry
#
# Regression coverage for the genie3 API-drift bug: the hotspot/length variant
# problem JSONs live in a separate dataset tree with non-uniform per-target naming,
# and combined hotspot+length variants don't exist at all. See NEXT_STEPS.md.
# ---------------------------------------------------------------------------

class TestBuildSelection:
    def test_bare_selection_for_all_mode_zero_length(self):
        assert _build_selection("01_bhrf1", "all", 0) == "01_bhrf1"

    def test_longbinder_suffix_for_all_mode_length_60(self):
        assert _build_selection("01_bhrf1", "all", 60) == "01_bhrf1_longbinder"
        assert _build_selection("06_insulinr", "all", 60) == "06_insulinr_longbinder"

    def test_bhrf1_ablate_competitors(self):
        assert _build_selection("01_bhrf1", "ablate_competitors", 0) == "01_bhrf1_ablate_others"

    def test_bhrf1_missed_only(self):
        assert _build_selection("01_bhrf1", "missed_only", 0) == "01_bhrf1_only_B92"

    def test_insulinr_ablate_competitors_uses_target_specific_suffix(self):
        # InsulinR's ablation target is B83, not a generic "_ablate_others" -- this is
        # deliberate per-target science (find_missed_hotspots.py), not a naming bug.
        assert _build_selection("06_insulinr", "ablate_competitors", 0) == "06_insulinr_ablate_b83"

    def test_insulinr_missed_only_has_no_variant(self):
        # InsulinR's "never attempted" hotspots (B59, B91) were identified but never
        # turned into a problem file -- this must raise, not silently build a selection
        # string that resolves to nothing.
        with pytest.raises(NoDatasetVariant):
            _build_selection("06_insulinr", "missed_only", 0)

    def test_combined_hotspot_and_length_variant_has_no_variant(self):
        # No target has a problem file combining a non-"all" hotspot_mode with
        # length_delta=60 (e.g. no "01_bhrf1_ablate_others_longbinder").
        with pytest.raises(NoDatasetVariant):
            _build_selection("01_bhrf1", "ablate_competitors", 60)
        with pytest.raises(NoDatasetVariant):
            _build_selection("01_bhrf1", "missed_only", 60)

    def test_unrecognized_target_hotspot_combo_has_no_variant(self):
        with pytest.raises(NoDatasetVariant):
            _build_selection("99_unknown_target", "ablate_competitors", 0)


class TestNeedsAblationConfig:
    def test_bare_selection_does_not_need_ablation_config(self):
        assert _needs_ablation_config("all", 0) is False

    def test_any_hotspot_variant_needs_ablation_config(self):
        assert _needs_ablation_config("ablate_competitors", 0) is True
        assert _needs_ablation_config("missed_only", 0) is True

    def test_any_length_variant_needs_ablation_config(self):
        assert _needs_ablation_config("all", 60) is True


class TestLiveRewardModelConfigRouting:
    def test_auto_selects_base_config_for_bare_selection(self):
        model = LiveRewardModel(genie3_root=Path("/fake/genie3"))
        cfg = model._config_for("all", 0)
        assert cfg == Path("/fake/genie3/branching/configs/experiment_trajectory_branching.yaml")

    def test_auto_selects_ablation_config_for_variant(self):
        model = LiveRewardModel(genie3_root=Path("/fake/genie3"))
        cfg = model._config_for("ablate_competitors", 0)
        assert cfg == Path(
            "/fake/genie3/branching/hotspot_ablation/configs/experiment_hotspot_ablation.yaml"
        )

    def test_explicit_override_always_wins(self):
        override = Path("/custom/my_config.yaml")
        model = LiveRewardModel(genie3_root=Path("/fake/genie3"), config_yaml=override)
        assert model._config_for("all", 0) == override
        assert model._config_for("missed_only", 60) == override

    def test_dataset_dir_matches_config_routing(self):
        """_dataset_dir_for (used for eval.py's --dataset-dir) must route the same way
        as _config_for (used for branching_wrapper.py's --config) -- eval.py is a
        separate genie3 script with its own dataset lookup, but both steps of a
        LiveRewardModel call need to agree on which dataset tree a given cell lives in."""
        model = LiveRewardModel(genie3_root=Path("/fake/genie3"))
        assert model._dataset_dir_for("all", 0) == Path(
            "/fake/genie3/data/design/binder_design/binderbench"
        )
        assert model._dataset_dir_for("ablate_competitors", 0) == Path(
            "/fake/genie3/branching/hotspot_ablation/dataset"
        )
        assert model._dataset_dir_for("all", 60) == Path(
            "/fake/genie3/branching/hotspot_ablation/dataset"
        )


# ---------------------------------------------------------------------------
# Interface geometry: hotspot_coverage + iCS (NEXT_STEPS.md §1.7/§1.8)
# ---------------------------------------------------------------------------

def _atom_line(serial, atom, resname, chain, resnum, x, y, z):
    """One PDB ATOM record in fixed-column format."""
    return (
        f"ATOM  {serial:>5} {atom:<4} {resname} {chain}{resnum:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00 50.00           C  \n"
    )


def _write_pdb(path, atoms):
    """atoms: list of (atom_name, resname, chain, resnum, x, y, z)."""
    with open(path, "w") as fh:
        for i, (atom, resname, chain, resnum, x, y, z) in enumerate(atoms, 1):
            fh.write(_atom_line(i, atom, resname, chain, resnum, x, y, z))


class TestParsePdbCbCoords:
    def test_prefers_cb_falls_back_to_ca_for_glycine(self, tmp_path):
        pdb = tmp_path / "t.pdb"
        _write_pdb(pdb, [
            ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
            ("CB", "ALA", "A", 1, 1.0, 0.0, 0.0),   # ALA has CB -> use CB
            ("CA", "GLY", "A", 2, 5.0, 0.0, 0.0),   # GLY has no CB -> fall back to CA
        ])
        coords = _parse_pdb_cb_coords(pdb)
        assert coords[("A", 1)][0] == pytest.approx(1.0)   # CB, not CA
        assert coords[("A", 2)][0] == pytest.approx(5.0)   # CA fallback


class TestHotspotCoverage:
    def _binder_target_pdb(self, tmp_path, binder_xyz, target_residues):
        """binder_xyz: list of binder CB coords. target_residues: {resnum: (x,y,z)}."""
        atoms = [("CB", "ALA", "A", i + 1, *xyz) for i, xyz in enumerate(binder_xyz)]
        atoms += [("CB", "ALA", "B", rn, *xyz) for rn, xyz in target_residues.items()]
        pdb = tmp_path / "bt.pdb"
        _write_pdb(pdb, atoms)
        return pdb

    def test_full_coverage_when_all_hotspots_contacted(self, tmp_path):
        # one binder atom at origin; two hotspots both within 8Å
        pdb = self._binder_target_pdb(
            tmp_path, [(0.0, 0.0, 0.0)], {64: (3.0, 0.0, 0.0), 73: (5.0, 0.0, 0.0)}
        )
        assert _compute_hotspot_coverage(pdb, ["B64", "B73"]) == pytest.approx(1.0)

    def test_partial_coverage(self, tmp_path):
        # B64 within 8Å, B92 far away (20Å)
        pdb = self._binder_target_pdb(
            tmp_path, [(0.0, 0.0, 0.0)], {64: (3.0, 0.0, 0.0), 92: (20.0, 0.0, 0.0)}
        )
        assert _compute_hotspot_coverage(pdb, ["B64", "B92"]) == pytest.approx(0.5)

    def test_none_when_no_binder_chain(self, tmp_path):
        pdb = self._binder_target_pdb(tmp_path, [], {64: (3.0, 0.0, 0.0)})
        assert _compute_hotspot_coverage(pdb, ["B64"]) is None

    def test_none_when_no_hotspot_present_in_structure(self, tmp_path):
        # hotspot B999 not in the PDB -> no valid hotspots -> None (not a misleading 0.0)
        pdb = self._binder_target_pdb(tmp_path, [(0.0, 0.0, 0.0)], {64: (3.0, 0.0, 0.0)})
        assert _compute_hotspot_coverage(pdb, ["B999"]) is None

    def test_cutoff_boundary(self, tmp_path):
        # exactly at 8Å counts as covered (<=)
        pdb = self._binder_target_pdb(tmp_path, [(0.0, 0.0, 0.0)], {64: (8.0, 0.0, 0.0)})
        assert _compute_hotspot_coverage(pdb, ["B64"], cutoff=8.0) == pytest.approx(1.0)
        assert _compute_hotspot_coverage(pdb, ["B64"], cutoff=7.9) is None or \
               _compute_hotspot_coverage(pdb, ["B64"], cutoff=7.9) == pytest.approx(0.0)


class TestInterfaceContactPairs:
    def test_finds_pairs_within_cutoff(self, tmp_path):
        atoms = [
            ("CB", "ALA", "A", 1, 0.0, 0.0, 0.0),
            ("CB", "ALA", "A", 2, 100.0, 0.0, 0.0),   # far from everything
            ("CB", "ALA", "B", 10, 3.0, 0.0, 0.0),    # within 8Å of binder res 1
            ("CB", "ALA", "B", 11, 50.0, 0.0, 0.0),   # far
        ]
        pdb = tmp_path / "c.pdb"
        _write_pdb(pdb, atoms)
        pairs = _interface_contact_pairs(pdb)
        assert (1, 10) in pairs
        assert (2, 10) not in pairs
        assert (1, 11) not in pairs


class TestComputeIcs:
    def test_low_pae_at_contacts_gives_high_ics(self):
        import numpy as np
        binder_len = 5
        n = 10
        contacts = [(1, 1), (2, 2)]   # (binder_resnum, target_resnum)
        pae = np.full((n, n), 25.0, dtype=np.float32)
        for b, t in contacts:
            i, j = b - 1, binder_len + (t - 1)
            pae[i, j] = pae[j, i] = 2.0
        ics = _compute_ics(pae, contacts, binder_len)
        assert ics == pytest.approx(np.exp(-2.0 / 10.0), abs=1e-4)

    def test_high_pae_gives_low_ics(self):
        import numpy as np
        binder_len = 5
        pae = np.full((10, 10), 25.0, dtype=np.float32)
        ics = _compute_ics(pae, [(1, 1)], binder_len)
        assert ics == pytest.approx(np.exp(-25.0 / 10.0), abs=1e-4)

    def test_empty_contacts_returns_none(self):
        import numpy as np
        assert _compute_ics(np.zeros((10, 10)), [], binder_len=5) is None

    def test_symmetrizes_pae(self):
        import numpy as np
        binder_len = 5
        pae = np.full((10, 10), 25.0, dtype=np.float32)
        i, j = 0, 5  # binder res 1, target res 1
        pae[i, j] = 2.0
        pae[j, i] = 8.0   # asymmetric -> should use mean 5.0
        ics = _compute_ics(pae, [(1, 1)], binder_len)
        assert ics == pytest.approx(np.exp(-5.0 / 10.0), abs=1e-4)

    def test_out_of_range_pairs_skipped(self):
        import numpy as np
        # target resnum maps past the matrix -> skipped, not an index error
        pae = np.full((10, 10), 2.0, dtype=np.float32)
        assert _compute_ics(pae, [(1, 999)], binder_len=5) is None
