"""Tests for Stage 5 FrontierBuffer."""

import numpy as np
import pytest

from buffer.frontier_buffer import FrontierBuffer, FrontierEntry, _ca_rmsd, _is_novel


def _make_entry(target, reward, x_T=None, timestep=800):
    if x_T is None:
        x_T = np.zeros((10, 3), dtype=np.float32)
    return FrontierEntry(
        x_T=x_T,
        target=target,
        levers={"timestep": timestep, "hotspot_mode": "all", "length_delta": 0},
        reward=reward,
    )


class TestCaRmsd:
    def test_identical_arrays_zero(self):
        a = np.ones((5, 3), dtype=np.float32)
        assert _ca_rmsd(a, a) == pytest.approx(0.0)

    def test_different_lengths_uses_min(self):
        a = np.zeros((5, 3), dtype=np.float32)
        b = np.ones((3, 3), dtype=np.float32)
        # only first 3 rows compared; all differ by sqrt(3) per row
        expected = float(np.sqrt(np.mean(3 * np.ones(3))))
        assert _ca_rmsd(a, b) == pytest.approx(expected, rel=1e-4)


class TestIsNovel:
    def test_empty_existing_is_novel(self):
        x = np.zeros((5, 3), dtype=np.float32)
        assert _is_novel(x, [])

    def test_identical_not_novel(self):
        x = np.zeros((5, 3), dtype=np.float32)
        e = _make_entry("t", 0.5, x_T=x)
        assert not _is_novel(x, [e])

    def test_far_entry_is_novel(self):
        x = np.zeros((5, 3), dtype=np.float32)
        y = np.ones((5, 3), dtype=np.float32) * 100  # very far away
        e = _make_entry("t", 0.5, x_T=x)
        assert _is_novel(y, [e])


class TestFrontierBuffer:
    def test_initialize_clears(self):
        buf = FrontierBuffer(size=4, seed=0)
        buf.initialize(["A", "B"])
        e = _make_entry("A", 0.5, x_T=np.zeros((5, 3), dtype=np.float32))
        buf.update(e)
        assert buf.size_for("A") == 1
        buf.initialize(["A", "B"])
        assert buf.size_for("A") == 0

    def test_update_accepts_novel_entry(self):
        buf = FrontierBuffer(size=4, seed=0)
        buf.initialize(["A"])
        e = _make_entry("A", 0.5, x_T=np.zeros((5, 3), dtype=np.float32))
        accepted = buf.update(e)
        assert accepted
        assert buf.size_for("A") == 1

    def test_update_rejects_duplicate(self):
        buf = FrontierBuffer(size=4, novelty_threshold=2.0, seed=0)
        buf.initialize(["A"])
        x = np.zeros((5, 3), dtype=np.float32)
        buf.update(_make_entry("A", 0.5, x_T=x))
        # same seed x_T = near-identical: rejected
        accepted = buf.update(_make_entry("A", 0.9, x_T=x + 0.01))
        assert not accepted

    def test_update_evicts_worst_when_full(self):
        buf = FrontierBuffer(size=2, novelty_threshold=0.0, seed=0)
        buf.initialize(["A"])
        x1 = np.zeros((5, 3), dtype=np.float32)
        x2 = np.ones((5, 3), dtype=np.float32) * 10
        buf.update(_make_entry("A", 0.3, x_T=x1))
        buf.update(_make_entry("A", 0.5, x_T=x2))
        # buffer full; add higher-reward entry with novel x_T
        x3 = np.ones((5, 3), dtype=np.float32) * 20
        accepted = buf.update(_make_entry("A", 0.7, x_T=x3))
        assert accepted
        assert buf.size_for("A") == 2
        # worst (0.3) should be gone
        rewards = {e.reward for e in buf._entries["A"]}
        assert 0.3 not in rewards

    def test_update_rejects_if_worse_than_all(self):
        buf = FrontierBuffer(size=2, novelty_threshold=0.0, seed=0)
        buf.initialize(["A"])
        x1 = np.zeros((5, 3), dtype=np.float32)
        x2 = np.ones((5, 3), dtype=np.float32) * 10
        buf.update(_make_entry("A", 0.5, x_T=x1))
        buf.update(_make_entry("A", 0.6, x_T=x2))
        x3 = np.ones((5, 3), dtype=np.float32) * 20
        accepted = buf.update(_make_entry("A", 0.2, x_T=x3))
        assert not accepted

    def test_top_returns_best_under_epsilon_1(self):
        buf = FrontierBuffer(size=4, epsilon=1.0, seed=0)  # always greedy
        buf.initialize(["A"])
        for i, reward in enumerate([0.3, 0.9, 0.5]):
            x = np.ones((5, 3), dtype=np.float32) * i * 10
            buf.update(_make_entry("A", reward, x_T=x))
        entry = buf.top("A")
        assert entry is not None
        assert entry.reward == 0.9

    def test_top_returns_none_for_empty(self):
        buf = FrontierBuffer(seed=0)
        buf.initialize(["A"])
        assert buf.top("A") is None

    def test_top_increments_visit_count(self):
        buf = FrontierBuffer(size=4, epsilon=1.0, seed=0)
        buf.initialize(["A"])
        x = np.zeros((5, 3), dtype=np.float32)
        buf.update(_make_entry("A", 0.5, x_T=x))
        buf.top("A")
        buf.top("A")
        assert buf._entries["A"][0].visit_count == 2

    def test_sample_seed_fresh_noise_when_empty(self):
        buf = FrontierBuffer(seed=42)
        buf.initialize(["A"])
        seed_arr = buf.sample_seed("A", shape=(5, 3))
        assert seed_arr.shape == (5, 3)
        assert not np.all(seed_arr == 0)

    def test_sample_seed_from_buffer(self):
        buf = FrontierBuffer(size=4, epsilon=1.0, seed=0)
        buf.initialize(["A"])
        x = np.ones((5, 3), dtype=np.float32) * 5.0
        buf.update(_make_entry("A", 0.8, x_T=x))
        seed_arr = buf.sample_seed("A", shape=(5, 3), perturb_scale=0.0)
        np.testing.assert_allclose(seed_arr, x, atol=1e-5)

    def test_stats(self):
        buf = FrontierBuffer(size=4, novelty_threshold=0.0, seed=0)
        buf.initialize(["A"])
        for i, r in enumerate([0.3, 0.7]):
            x = np.ones((5, 3), dtype=np.float32) * i * 10
            buf.update(_make_entry("A", r, x_T=x))
        s = buf.stats()
        assert "A" in s
        assert s["A"]["n"] == 2
        assert s["A"]["best_reward"] == pytest.approx(0.7)

    def test_len(self):
        buf = FrontierBuffer(size=4, novelty_threshold=0.0, seed=0)
        buf.initialize(["A", "B"])
        for i in range(3):
            x = np.ones((5, 3), dtype=np.float32) * i * 10
            buf.update(_make_entry("A", 0.5 + i * 0.1, x_T=x))
        assert len(buf) == 3
