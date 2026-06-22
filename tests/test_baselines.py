"""Tests for the bandit (the load-bearing baseline) and the simple policies."""

from __future__ import annotations

import random

import numpy as np

from baselines.contextual_bandit import ContextualBandit
from baselines.fixed_heuristic import FixedHeuristic
from baselines.random_policy import RandomPolicy
from envs.genie_branch_env import GenieBranchEnv
from oracle.reward_oracle import OfflineRewardModel


def _biased_records():
    """Construct data where, per target, ONE timestep is clearly best -- and the best
    timestep DIFFERS by target.  A contextual learner must recover the per-target optimum."""
    recs, cid = [], 0
    best_ts = {"01_bhrf1": 800, "06_insulinr": 900}
    for target in best_ts:
        for ts in (700, 800, 900):
            success_p = 0.9 if ts == best_ts[target] else 0.05
            for _ in range(30):
                recs.append({
                    "target": target, "branch_timestep": ts, "hotspot_mode": "all",
                    "length_delta": 0, "binder_length": 100,
                    "iptm": 0.9 if random.random() < success_p else 0.4,
                    "avg_interface_pae": 3.0 if random.random() < success_p else 18.0,
                    "complex_success": random.random() < success_p,
                    "diversity": 0.5, "binder_seq": "AAAA", "child_id": cid,
                    "source_sweep": "fixture",
                })
                cid += 1
    return recs


def _env(seed=0):
    oracle = OfflineRewardModel(records=_biased_records(), rng=random.Random(seed))
    return GenieBranchEnv(oracle=oracle, levers=("timestep",), seed=seed)


def test_random_policy_interface():
    env = _env()
    pol = RandomPolicy(env.action_space.n, seed=0)
    a = pol.select(None, "01_bhrf1")
    assert 0 <= a < env.action_space.n
    pol.update("01_bhrf1", a, 1.0)  # no-op, must not raise


def test_fixed_heuristic_picks_a_valid_action():
    env = _env()
    fixed = FixedHeuristic(env)
    assert 0 <= fixed.best_action < env.action_space.n


def test_bandit_learns_per_target_optimum():
    random.seed(0)
    env = _env(seed=0)
    bandit = ContextualBandit(env, exploration="ucb", seed=0)
    # train online
    obs, info = env.reset(seed=0)
    for _ in range(3000):
        t = info["target"]
        a = bandit.select(obs, t)
        obs, r, term, trunc, info = env.step(a)
        bandit.update(t, a, float(r))
        obs, info = env.reset()
    table = bandit.policy_table()
    # bandit should pick the target-specific best timestep
    assert env.decode_action(table["01_bhrf1"])["timestep"] == 800
    assert env.decode_action(table["06_insulinr"])["timestep"] == 900


def test_bandit_warm_start_runs():
    env = _env()
    bandit = ContextualBandit(env, warm_start=True, seed=0)
    a = bandit.select(None, "01_bhrf1")
    assert 0 <= a < env.action_space.n
