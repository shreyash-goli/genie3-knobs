"""Tests for CommitmentWindowDetector and the windowed DiffusionInterventionEnv."""

import numpy as np
import pytest

from envs.commitment_window import (
    COMMIT_ACTION,
    CommitmentWindowDetector,
    DiffusionInterventionEnv,
    HOTSPOT_MODES,
    LENGTH_DELTAS,
    N_WINDOW_STEPS,
    _timestep_schedule,
)


# ---------------------------------------------------------------------------
# CommitmentWindowDetector
# ---------------------------------------------------------------------------

def _make_records(target: str, timesteps: list[int], variance_shape: str) -> list[dict]:
    """Build synthetic records with controlled per-timestep reward variance.

    variance_shape:
        "peak_at_mid"  : variance rises then falls (realistic commitment window)
        "flat"         : uniform variance across all timesteps
    """
    import random
    rng = random.Random(42)
    records = []
    n_children = 10
    for ts in timesteps:
        if variance_shape == "peak_at_mid":
            mid = timesteps[len(timesteps) // 2]
            spread = 0.3 * (1 - abs(ts - mid) / (max(timesteps) - min(timesteps) + 1))
        else:
            spread = 0.1
        for _ in range(n_children):
            iptm = max(0.0, min(1.0, 0.5 + rng.gauss(0, spread)))
            records.append({
                "target": target,
                "branch_timestep": ts,
                "hotspot_mode": "all",
                "length_delta": 0,
                "iptm": iptm,
                "avg_interface_pae": 10.0,
                "complex_success": iptm > 0.7,
            })
    return records


class TestCommitmentWindowDetector:
    def test_detects_peak_at_middle(self):
        timesteps = [700, 750, 800, 850, 900]
        records = _make_records("t1", timesteps, "peak_at_mid")
        detector = CommitmentWindowDetector(min_n=3)
        windows = detector.detect(records)
        assert "t1" in windows
        w = windows["t1"]
        assert w.peak_variance_ts in [750, 800, 850]

    def test_returns_window_with_correct_fields(self):
        records = _make_records("t2", [700, 800, 900], "flat")
        detector = CommitmentWindowDetector(min_n=3)
        windows = detector.detect(records)
        w = windows["t2"]
        assert w.target == "t2"
        assert w.window_start <= w.peak_variance_ts <= w.window_end
        assert all(ts in w.variance_by_ts for ts in [700, 800, 900])

    def test_skips_target_with_insufficient_data(self):
        records = [
            {"target": "sparse", "branch_timestep": 800, "hotspot_mode": "all",
             "length_delta": 0, "iptm": 0.6, "complex_success": False},
        ]
        detector = CommitmentWindowDetector(min_n=3)
        windows = detector.detect(records)
        assert "sparse" not in windows

    def test_multiple_targets_independent(self):
        r1 = _make_records("alpha", [700, 800, 900], "peak_at_mid")
        r2 = _make_records("beta", [700, 800, 900], "flat")
        detector = CommitmentWindowDetector(min_n=3)
        windows = detector.detect(r1 + r2)
        assert "alpha" in windows
        assert "beta" in windows


# ---------------------------------------------------------------------------
# _timestep_schedule helper
# ---------------------------------------------------------------------------

class TestTimestepSchedule:
    def test_length(self):
        sched = _timestep_schedule(700, 950, 10)
        assert len(sched) == 10

    def test_endpoints(self):
        sched = _timestep_schedule(700, 950, 10)
        assert sched[0] == 700
        assert sched[-1] == 950

    def test_monotone(self):
        sched = _timestep_schedule(700, 950, 10)
        assert all(sched[i] <= sched[i + 1] for i in range(len(sched) - 1))

    def test_single_step(self):
        sched = _timestep_schedule(800, 900, 1)
        assert sched == [800]


# ---------------------------------------------------------------------------
# DiffusionInterventionEnv — windowed MDP (offline mode only, no genie3 needed)
# ---------------------------------------------------------------------------

class TestDiffusionInterventionEnv:
    def _make_env(self, targets=None, seed=0):
        return DiffusionInterventionEnv(
            targets=targets or ["01_bhrf1", "06_insulinr"],
            oracle_mode="offline",
            seed=seed,
        )

    def test_obs_shape_and_dtype(self):
        env = self._make_env(["01_bhrf1", "06_insulinr"])
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

    def test_seeded_oracle_is_reproducible(self):
        """Two same-seed envs must produce identical reward sequences for the same actions
        (offline oracle RNG is seeded from the env seed). Without this the oracle draws a
        random logged child per cell, adding a large noise floor (see §6)."""
        def rollout(seed):
            env = self._make_env(["01_bhrf1", "06_insulinr"], seed=seed)
            env.reset(seed=seed)
            rewards = []
            done = False
            while not done:
                _, r, term, trunc, _ = env.step(0)
                rewards.append(r)
                done = term or trunc
            return rewards

        assert rollout(123) == rollout(123)          # identical seed -> identical rewards

    def test_mask_target_onehot_zeros_the_onehot_block(self):
        n_targets = 2
        env = DiffusionInterventionEnv(
            targets=["01_bhrf1", "06_insulinr"], oracle_mode="offline",
            mask_target_onehot=True, seed=0,
        )
        obs, _ = env.reset(options={"target": "01_bhrf1"})
        # the target one-hot block (first n_targets entries) must be all-zero
        assert np.allclose(obs[:n_targets], 0.0)

    def test_action_space_includes_commit(self):
        # HOTSPOT_MODES conditioning arms + 1 commit action (§3.2).
        env = self._make_env()
        assert env.action_space.n == len(HOTSPOT_MODES) + 1
        assert COMMIT_ACTION == len(HOTSPOT_MODES)

    def test_n_actions_property(self):
        env = self._make_env()
        assert env.n_actions == len(HOTSPOT_MODES) + 1

    def test_episode_runs_n_window_steps(self):
        env = self._make_env()
        env.reset(seed=0)
        for step_i in range(N_WINDOW_STEPS - 1):
            _, reward, terminated, truncated, _ = env.step(0)
            assert not terminated, f"terminated early at step {step_i}"
            assert reward == 0.0, "intermediate step should have zero reward"
        _, reward, terminated, truncated, _ = env.step(0)
        assert terminated
        assert truncated is False
        assert reward > 0.0 or reward == 0.0  # reward is a float (may be 0 for failed episode)
        assert isinstance(reward, float)

    def test_terminal_reward_is_nonzero_on_success(self):
        # Run many episodes until we get at least one non-zero terminal reward.
        env = self._make_env()
        rewards = []
        for ep in range(10):
            env.reset(seed=ep)
            for _ in range(N_WINDOW_STEPS - 1):
                env.step(0)
            _, r, _, _, _ = env.step(0)
            rewards.append(r)
        assert any(r > 0 for r in rewards), "expected at least one non-zero terminal reward"

    def test_intermediate_steps_zero_reward(self):
        env = self._make_env()
        env.reset(seed=1)
        for _ in range(N_WINDOW_STEPS - 1):
            _, reward, terminated, _, _ = env.step(1)
            assert reward == 0.0
            assert not terminated

    def test_reset_with_target_option(self):
        env = self._make_env(["01_bhrf1", "06_insulinr"])
        obs, info = env.reset(options={"target": "01_bhrf1"})
        assert info["target"] == "01_bhrf1"
        # one-hot index 0 should be 1
        assert obs[0] == pytest.approx(1.0)
        assert obs[1] == pytest.approx(0.0)

    def test_reset_with_length_delta_option(self):
        env = self._make_env()
        _, info = env.reset(options={"length_delta": 60})
        assert info["length_delta"] == 60

    def test_length_delta_in_valid_set(self):
        env = self._make_env()
        for _ in range(20):
            _, info = env.reset()
            assert info["length_delta"] in LENGTH_DELTAS

    def test_timestep_schedule_in_info(self):
        env = self._make_env()
        _, info = env.reset()
        sched = info["timestep_schedule"]
        assert len(sched) == N_WINDOW_STEPS
        assert sched[0] <= sched[-1]

    def test_step_info_contains_hotspot_mode(self):
        env = self._make_env()
        env.reset(seed=0)
        for i, mode in enumerate(HOTSPOT_MODES):
            env.reset(seed=i)
            for _ in range(N_WINDOW_STEPS - 1):
                env.step(i % len(HOTSPOT_MODES))
            _, _, _, _, info = env.step(i % len(HOTSPOT_MODES))
            assert info["hotspot_mode"] in HOTSPOT_MODES

    def test_decode_action_returns_hotspot_mode(self):
        env = self._make_env()
        for i, mode in enumerate(HOTSPOT_MODES):
            assert env.decode_action(i) == {"hotspot_mode": mode}

    def test_all_actions_complete_episode(self):
        env = self._make_env()
        for action in range(len(HOTSPOT_MODES)):
            env.reset(seed=action)
            for _ in range(N_WINDOW_STEPS - 1):
                env.step(action)
            _, _, terminated, _, _ = env.step(action)
            assert terminated

    def test_obs_step_progress_increases(self):
        env = self._make_env(["01_bhrf1"])
        n_targets = len(env.targets)
        obs, _ = env.reset(seed=0)
        prev_progress = obs[n_targets]  # step_progress is first context element
        for _ in range(N_WINDOW_STEPS - 1):
            obs, _, _, _, _ = env.step(0)
            cur_progress = obs[n_targets]
            assert cur_progress >= prev_progress
            prev_progress = cur_progress

    # -- commit action (§3.2) ----------------------------------------------

    def test_commit_terminates_early_with_terminal_reward(self):
        env = self._make_env(["01_bhrf1"])
        env.reset(seed=0)
        env.step(0)  # one intermediate step (not terminal)
        _, reward, terminated, truncated, info = env.step(COMMIT_ACTION)
        assert terminated
        assert truncated is False
        assert info["termination_reason"] == "commit"
        assert info["commit"] is True
        assert info["step"] == 2  # ended at step 2, well before N_WINDOW_STEPS
        assert isinstance(reward, float)

    def test_commit_reward_matches_unscaled_terminal_reward(self):
        # A commit must pay the full (unscaled) terminal reward, not the
        # intermediate_reward_scale-shaped one.
        from oracle.reward_oracle import compute_reward
        env = DiffusionInterventionEnv(
            targets=["01_bhrf1"], oracle_mode="offline",
            intermediate_reward_scale=0.1, seed=3,
        )
        env.reset(seed=3)
        _, reward, terminated, _, info = env.step(COMMIT_ACTION)
        assert terminated
        assert reward == pytest.approx(compute_reward(info))

    def test_timeout_reason_when_window_elapses(self):
        env = self._make_env(["01_bhrf1"])
        env.reset(seed=0)
        info = None
        for _ in range(N_WINDOW_STEPS):
            _, _, terminated, _, info = env.step(0)
        assert terminated
        assert info["termination_reason"] == "timeout"
        assert info["commit"] is False

    def test_commit_at_step_zero_is_allowed(self):
        env = self._make_env(["01_bhrf1"])
        env.reset(seed=0)
        _, _, terminated, _, info = env.step(COMMIT_ACTION)
        assert terminated
        assert info["termination_reason"] == "commit"
        assert info["step"] == 1

    def test_decode_action_commit(self):
        env = self._make_env()
        assert env.decode_action(COMMIT_ACTION) == {"commit": True}

    # -- action-history observation (§3.2/§3.3) ----------------------------

    def test_obs_action_history_counts(self):
        env = self._make_env(["01_bhrf1"])
        n_targets = len(env.targets)
        n_modes = len(HOTSPOT_MODES)
        obs, _ = env.reset(seed=0)
        # history block is the final n_modes entries, zero at reset
        assert np.allclose(obs[-n_modes:], 0.0)
        env.step(0)
        env.step(0)
        obs, _, _, _, _ = env.step(1)
        history = obs[-n_modes:]
        assert history[0] == pytest.approx(2.0 / N_WINDOW_STEPS)
        assert history[1] == pytest.approx(1.0 / N_WINDOW_STEPS)
        assert history[2] == pytest.approx(0.0)

    def test_commit_does_not_increment_action_history(self):
        env = self._make_env(["01_bhrf1"])
        n_modes = len(HOTSPOT_MODES)
        env.reset(seed=0)
        env.step(0)
        obs, _, _, _, _ = env.step(COMMIT_ACTION)
        history = obs[-n_modes:]
        # only the single hotspot step counts; commit does not
        assert history[0] == pytest.approx(1.0 / N_WINDOW_STEPS)

    def test_obs_prefix_unchanged_by_history_block(self):
        # The one-hot block and step_progress must keep their original indices, so
        # existing consumers/tests that index obs[:n_targets] / obs[n_targets] still work.
        env = self._make_env(["01_bhrf1", "06_insulinr"])
        n_targets = len(env.targets)
        obs, _ = env.reset(seed=0, options={"target": "01_bhrf1"})
        assert obs[0] == pytest.approx(1.0)
        assert obs[1] == pytest.approx(0.0)
        assert obs[n_targets] == pytest.approx(0.0)  # step_progress at reset


# ---------------------------------------------------------------------------
# DiffusionInterventionEnv — live mode oracle call count (NEXT_STEPS.md section 0.2)
#
# Regression test for a bug where _query_oracle called the full LiveRewardModel on every
# one of the N_WINDOW_STEPS steps instead of once per episode (at the terminal step only),
# making a live episode cost ~10x its documented/budgeted time.
# ---------------------------------------------------------------------------

class _CountingLiveRewardModel:
    """Stand-in for oracle.live_oracle.LiveRewardModel that counts calls instead of
    shelling out to genie3."""

    call_count = 0

    def __init__(self, *args, **kwargs):
        pass

    def sample(self, target, timestep, hotspot_mode="all", length_delta=0):
        type(self).call_count += 1
        return {"iptm": 0.9, "avg_interface_pae": 5.0, "complex_success": True}, 0


class TestLiveModeOracleCallCount:
    def test_live_episode_calls_oracle_exactly_once(self, monkeypatch):
        monkeypatch.setattr(
            "oracle.live_oracle.LiveRewardModel", _CountingLiveRewardModel
        )
        _CountingLiveRewardModel.call_count = 0

        env = DiffusionInterventionEnv(
            targets=["01_bhrf1"], oracle_mode="live", seed=0,
        )
        env.reset(seed=0)
        done = False
        n_steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            n_steps += 1

        assert n_steps == N_WINDOW_STEPS
        assert _CountingLiveRewardModel.call_count == 1

    def test_non_terminal_live_steps_get_zero_reward(self, monkeypatch):
        monkeypatch.setattr(
            "oracle.live_oracle.LiveRewardModel", _CountingLiveRewardModel
        )
        _CountingLiveRewardModel.call_count = 0

        env = DiffusionInterventionEnv(
            targets=["01_bhrf1"], oracle_mode="live",
            intermediate_reward_scale=0.5, seed=0,
        )
        env.reset(seed=0)
        for _ in range(N_WINDOW_STEPS - 1):
            _, reward, terminated, _, _ = env.step(0)
            assert reward == pytest.approx(0.0)
            assert not terminated
        _, terminal_reward, terminated, _, _ = env.step(0)
        assert terminated
        assert terminal_reward > 0.0

    def test_commit_fires_the_single_live_oracle_call(self, monkeypatch):
        # A commit ends the episode early; the one-call-per-episode invariant must hold,
        # with the call fired at the commit step (not step N-1).
        monkeypatch.setattr(
            "oracle.live_oracle.LiveRewardModel", _CountingLiveRewardModel
        )
        _CountingLiveRewardModel.call_count = 0

        env = DiffusionInterventionEnv(
            targets=["01_bhrf1"], oracle_mode="live", seed=0,
        )
        env.reset(seed=0)
        _, reward, terminated, _, info = env.step(0)   # intermediate: no oracle call
        assert not terminated
        assert _CountingLiveRewardModel.call_count == 0
        _, reward, terminated, _, info = env.step(COMMIT_ACTION)
        assert terminated
        assert info["termination_reason"] == "commit"
        assert _CountingLiveRewardModel.call_count == 1
        assert reward > 0.0
