"""Tests for the Stage-0 ingester's pure parsing logic (no /pscratch dependency)."""

from __future__ import annotations

import json

from instrumentation.trajectory_logger import (
    TrajectoryLogger,
    TrajectoryRecord,
    _branch_timestep_from_dir,
    load_records,
    parse_problem_variant,
)


def test_parse_problem_variant_base():
    assert parse_problem_variant("01_bhrf1") == ("01_bhrf1", "all", 0)


def test_parse_problem_variant_longbinder():
    assert parse_problem_variant("01_bhrf1_longbinder") == ("01_bhrf1", "all", 60)


def test_parse_problem_variant_ablate():
    base, mode, ld = parse_problem_variant("01_bhrf1_ablate_others")
    assert (base, mode, ld) == ("01_bhrf1", "ablate_competitors", 0)


def test_parse_problem_variant_only():
    base, mode, ld = parse_problem_variant("01_bhrf1_only_B92")
    assert (base, mode, ld) == ("01_bhrf1", "missed_only", 0)


def test_parse_problem_variant_ablate_specific_residue():
    base, mode, ld = parse_problem_variant("06_insulinr_ablate_b83")
    assert (base, mode, ld) == ("06_insulinr", "ablate_competitors", 0)


def test_branch_timestep_parsing():
    assert _branch_timestep_from_dir("branch_t_800") == 800
    assert _branch_timestep_from_dir("branch_s_10") == 10
    assert _branch_timestep_from_dir("configs") is None


def test_logger_roundtrip(tmp_path):
    jsonl = tmp_path / "rec.jsonl"
    with TrajectoryLogger(jsonl_path=jsonl, mirror_sqlite=False) as logger:
        logger.log(TrajectoryRecord(target="01_bhrf1", branch_timestep=800,
                                    hotspot_mode="all", length_delta=0, iptm=0.88))
    rows = load_records(jsonl)
    assert len(rows) == 1
    assert rows[0]["target"] == "01_bhrf1"
    assert rows[0]["iptm"] == 0.88
