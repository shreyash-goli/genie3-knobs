"""Tests for Stage 6 CommitmentWindowDetector and DiffusionInterventionEnv."""

import numpy as np
import pytest

from envs.commitment_window import (
    CommitmentWindowDetector,
    DiffusionInterventionEnv,
    INTERVENTION_SCALES,
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
            # variance proportional to proximity to mid
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
        # peak should be at or near 800 (the middle timestep)
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
        # Only 1 record per timestep — below min_n=3
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
# DiffusionInterventionEnv (offline mode only — no genie3 needed)
# ---------------------------------------------------------------------------

class TestDiffusionInterventionEnv:
    def _make_env(self, targets=None):
        return DiffusionInterventionEnv(
            targets=targets or ["01_bhrf1", "06_insulinr"],
            oracle_mode="offline",
            seed=0,
        )

    def test_obs_shape(self):
        env = self._make_env(["01_bhrf1", "06_insulinr"])
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

    def test_action_space_size(self):
        env = self._make_env()
        assert env.action_space.n == len(INTERVENTION_SCALES)

    def test_step_returns_scalar_reward(self):
        env = self._make_env()
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(2)  # scale=1.0
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.5  # generous upper bound
        assert terminated is True
        assert truncated is False

    def test_reset_with_target_option(self):
        env = self._make_env(["01_bhrf1", "06_insulinr"])
        obs, info = env.reset(options={"target": "01_bhrf1"})
        assert info["target"] == "01_bhrf1"
        # one-hot should have a 1 at position 0
        n_targets = len(env.targets)
        assert obs[0] == pytest.approx(1.0)
        assert obs[1] == pytest.approx(0.0)

    def test_decode_action_returns_correct_scale(self):
        env = self._make_env()
        for i, scale in enumerate(INTERVENTION_SCALES):
            assert env.decode_action(i)["direction_scale"] == scale

    def test_episode_is_one_step(self):
        env = self._make_env()
        env.reset(seed=0)
        _, _, terminated, truncated, _ = env.step(0)
        assert terminated

    def test_n_actions_property(self):
        env = self._make_env()
        assert env.n_actions == len(INTERVENTION_SCALES)
