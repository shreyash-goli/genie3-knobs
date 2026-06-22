"""Env smoke tests: step() returns valid (obs, reward, terminated, truncated, info).

Uses a tiny in-memory fixture dataset so the tests do not depend on the /pscratch sweeps."""

from __future__ import annotations

import random

import numpy as np
import pytest

from envs.genie_branch_env import GenieBranchEnv
from oracle.reward_oracle import OfflineRewardModel


def _fixture_records():
    recs = []
    cid = 0
    for target in ("01_bhrf1", "06_insulinr"):
        for ts in (700, 800, 900):
            for mode, ld in (("all", 0), ("all", 60), ("ablate_competitors", 0)):
                for _ in range(5):
                    recs.append({
                        "target": target, "branch_timestep": ts, "hotspot_mode": mode,
                        "length_delta": ld, "binder_length": 120,
                        "iptm": random.uniform(0.5, 0.95),
                        "avg_interface_pae": random.uniform(2, 15),
                        "complex_success": random.random() < 0.3,
                        "hotspot_coverage": random.uniform(0, 1),
                        "diversity": random.uniform(0, 1),
                        "binder_seq": "ACDEFG", "child_id": cid,
                        "source_sweep": "fixture",
                    })
                    cid += 1
    return recs


@pytest.fixture
def oracle():
    return OfflineRewardModel(records=_fixture_records(), rng=random.Random(0))


def test_step_shapes_timestep_only(oracle):
    env = GenieBranchEnv(oracle=oracle, levers=("timestep",), seed=0)
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert env.action_space.n == 3  # 3 timesteps
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert terminated is True and truncated is False
    assert "backoff" in info and "target" in info


def test_step_shapes_full_levers(oracle):
    env = GenieBranchEnv(oracle=oracle, levers=("timestep", "length", "hotspot"), seed=0)
    # 3 timesteps x 2 lengths x 2 modes = 12 (intersected with what's in the fixture)
    assert env.action_space.n == 12
    obs, info = env.reset()
    for a in range(env.action_space.n):
        env.reset()
        obs, reward, term, trunc, info = env.step(a)
        assert 0.0 <= reward <= 1.0


def test_decode_action_roundtrip(oracle):
    env = GenieBranchEnv(oracle=oracle, levers=("timestep", "length", "hotspot"), seed=0)
    seen = {tuple(env.decode_action(a).values()) for a in range(env.action_space.n)}
    assert len(seen) == env.action_space.n  # all actions distinct


def test_backoff_reported(oracle):
    env = GenieBranchEnv(oracle=oracle, levers=("timestep", "length", "hotspot"), seed=0)
    env.reset(options={"target": "01_bhrf1"})
    _, _, _, _, info = env.step(0)
    assert info["backoff"] in {"exact", "drop_length", "drop_hotspot",
                               "nearest_timestep", "target_global"}
