"""Tests for the contextual bandit wired against the windowed MDP (NEXT_STEPS.md §3.1).

The bandit was originally built for the one-shot GenieBranchEnv (see tests/test_baselines.py).
Here we test it against DiffusionInterventionEnv via the explicit-kwargs compatibility path and
the `n_actions` restriction that excludes the commit action, plus the train/eval helpers in
experiments/ppo_vs_bandit_offline.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines.contextual_bandit import ContextualBandit
from envs.commitment_window import COMMIT_ACTION, DiffusionInterventionEnv, HOTSPOT_MODES
from oracle.reward_oracle import RewardWeights


TARGETS = ["01_bhrf1", "06_insulinr"]


def _env(seed=0):
    return DiffusionInterventionEnv(targets=TARGETS, oracle_mode="offline", seed=seed)


def _bandit(env, seed=0):
    # The explicit-kwargs path: DiffusionInterventionEnv has no public `oracle`/`reward_weights`,
    # so they must be supplied; n_actions restricts to the 3 hotspot arms (excludes commit).
    return ContextualBandit(
        env, exploration="ucb", warm_start=False,
        oracle=env._oracle, weights=RewardWeights(),
        n_actions=len(HOTSPOT_MODES), seed=seed,
    )


def test_constructs_against_windowed_env_without_attribute_error():
    env = _env()
    bandit = _bandit(env)
    assert bandit.n_actions == len(HOTSPOT_MODES)  # 3, not the env's Discrete(4)
    assert set(bandit.targets) == set(TARGETS)


def test_n_actions_override_excludes_commit():
    env = _env()
    bandit = _bandit(env)
    rng_targets = TARGETS * 50
    for i, target in enumerate(rng_targets):
        obs, _ = env.reset(seed=i, options={"target": target})
        a = bandit.select(obs, target)
        assert 0 <= a < len(HOTSPOT_MODES)
        assert a != COMMIT_ACTION
        bandit.update(target, a, 0.5)


def test_default_n_actions_uses_full_action_space():
    # Back-compat: without the override the bandit sees the full Discrete(4).
    env = _env()
    bandit = ContextualBandit(
        env, warm_start=False, oracle=env._oracle, weights=RewardWeights(), seed=0)
    assert bandit.n_actions == env.action_space.n == len(HOTSPOT_MODES) + 1


def test_greedy_action_and_policy_table_cover_all_targets():
    from experiments.ppo_vs_bandit_offline import _train_bandit
    env = _env()
    bandit = _bandit(env)
    _train_bandit(env, bandit, TARGETS, n_episodes=80)
    table = bandit.policy_table()
    assert set(table) == set(TARGETS)
    for target in TARGETS:
        a = bandit.greedy_action(target)
        assert 0 <= a < len(HOTSPOT_MODES)


def test_eval_closure_plays_valid_hotspot_actions():
    from experiments.ppo_vs_bandit_offline import _bandit_policy, _train_bandit, _eval_policy
    env = _env()
    bandit = _bandit(env)
    _train_bandit(env, bandit, TARGETS, n_episodes=80)
    rewards = _eval_policy(env, _bandit_policy(bandit, len(TARGETS)), n_episodes=20,
                           seed_offset=1000)
    assert len(rewards) == 20
    assert all(np.isfinite(r) for r in rewards)
    # offline terminal reward lives in a sane range
    assert all(-1.0 <= r <= 5.0 for r in rewards)


def test_bandit_policy_recovers_target_from_onehot():
    from experiments.ppo_vs_bandit_offline import _bandit_policy
    env = _env()
    bandit = _bandit(env)
    # force distinct greedy arms per target so we can tell them apart
    bandit._q["01_bhrf1"] = np.array([5.0, 0.0, 0.0])
    bandit._q["06_insulinr"] = np.array([0.0, 0.0, 5.0])
    policy = _bandit_policy(bandit, len(TARGETS))
    obs0, _ = env.reset(options={"target": "01_bhrf1"})
    obs1, _ = env.reset(options={"target": "06_insulinr"})
    assert policy(obs0) == 0
    assert policy(obs1) == 2
