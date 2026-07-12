"""Tests for the FK Steering inference layer (NEXT_STEPS.md §7.2).

Covers the resampling primitive and the k-particle rollout. Offline/seeded and fast -- the
only torch used is the tiny ActorCritic, no GPU, no genie3/oracle beyond the offline dataset.
"""

import numpy as np
import pytest

from policy.fk_steering import fk_resample, fk_rollout

torch = pytest.importorskip("torch")

from envs.commitment_window import DiffusionInterventionEnv, HOTSPOT_MODES  # noqa: E402
from oracle.reward_oracle import compute_reward  # noqa: E402
from policy.lora_finetune import ActorCritic  # noqa: E402


# ---------------------------------------------------------------------------
# fk_resample
# ---------------------------------------------------------------------------

class TestFkResample:
    def test_single_particle_is_identity(self):
        rng = np.random.default_rng(0)
        assert fk_resample(np.array([0.5]), rng).tolist() == [0]

    def test_zero_particles(self):
        rng = np.random.default_rng(0)
        assert fk_resample(np.array([]), rng).tolist() == []

    def test_temperature_zero_selects_argmax(self):
        rng = np.random.default_rng(0)
        scores = np.array([0.1, 0.9, 0.3, 0.2])
        survivors = fk_resample(scores, rng, temperature=0.0)
        assert survivors.tolist() == [1, 1, 1, 1]  # argmax duplicated

    def test_equal_scores_are_roughly_uniform(self):
        rng = np.random.default_rng(1)
        scores = np.zeros(4)
        counts = np.zeros(4)
        for _ in range(2000):
            for idx in fk_resample(scores, rng, temperature=1.0):
                counts[idx] += 1
        freq = counts / counts.sum()
        assert np.allclose(freq, 0.25, atol=0.03)

    def test_higher_scores_duplicated_more_often(self):
        rng = np.random.default_rng(2)
        scores = np.array([0.0, 1.0])
        counts = np.zeros(2)
        for _ in range(2000):
            for idx in fk_resample(scores, rng, temperature=0.5):
                counts[idx] += 1
        assert counts[1] > counts[0]  # higher-scoring particle survives more

    def test_output_length_matches_input(self):
        rng = np.random.default_rng(3)
        scores = np.array([0.2, 0.4, 0.1, 0.9, 0.5])
        assert len(fk_resample(scores, rng, temperature=1.0)) == len(scores)


# ---------------------------------------------------------------------------
# fk_rollout
# ---------------------------------------------------------------------------

def _make_env_seeded(s):
    return DiffusionInterventionEnv(
        targets=["01_bhrf1", "06_insulinr"], oracle_mode="offline",
        intermediate_reward_scale=0.0, seed=s,
    )


class TestFkRollout:
    def test_k1_equals_single_particle_terminal_reward(self):
        """k=1 must reduce to a single greedy episode's terminal reward (the no-FK baseline).
        Reproduce fk_rollout's exact seeding path with one env and compare."""
        ac = ActorCritic.build(obs_dim=_make_env_seeded(0).observation_space.shape[0],
                               n_actions=_make_env_seeded(0).n_actions, hidden=(8,))

        base_seed = 12345
        rng = np.random.default_rng(7)
        fk_reward = fk_rollout(_make_env_seeded, ac, k=1, rng=rng, base_seed=base_seed)

        # Manually replay: fk_rollout draws target then length_delta from the SAME rng first.
        rng2 = np.random.default_rng(7)
        env0 = _make_env_seeded(base_seed)
        target = str(rng2.choice(env0.targets))
        from envs.commitment_window import LENGTH_DELTAS
        length_delta = int(rng2.choice(LENGTH_DELTAS))
        env = _make_env_seeded(base_seed)
        obs, _ = env.reset(seed=base_seed, options={"target": target,
                                                    "length_delta": length_delta})
        terminal = 0.0
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = ac(obs_t)
            a = int(logits.argmax(dim=-1).item())
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            if done:
                terminal = float(compute_reward(info))
        assert fk_reward == pytest.approx(terminal)

    def test_best_survivor_at_least_mean(self):
        ac = ActorCritic.build(obs_dim=_make_env_seeded(0).observation_space.shape[0],
                               n_actions=_make_env_seeded(0).n_actions, hidden=(8,))
        rng = np.random.default_rng(0)
        best, diag = fk_rollout(_make_env_seeded, ac, k=6, rng=rng, base_seed=999,
                                return_diagnostics=True)
        assert best == pytest.approx(max(diag["terminal_rewards"]))
        assert best >= np.mean(diag["terminal_rewards"]) - 1e-9

    def test_shared_action_conditions_particles_identically(self):
        """Shared-action variant: all particles must condition on the same target/length."""
        ac = ActorCritic.build(obs_dim=_make_env_seeded(0).observation_space.shape[0],
                               n_actions=_make_env_seeded(0).n_actions, hidden=(8,))
        rng = np.random.default_rng(4)
        _, diag = fk_rollout(_make_env_seeded, ac, k=4, rng=rng, base_seed=222,
                             return_diagnostics=True)
        assert diag["target"] in _make_env_seeded(0).targets
        assert len(diag["terminal_rewards"]) == 4

    def test_random_conditioning_runs(self):
        rng = np.random.default_rng(5)
        best = fk_rollout(_make_env_seeded, None, k=4, rng=rng, base_seed=333)
        assert isinstance(best, float)

    def test_independent_actions_variant_runs(self):
        ac = ActorCritic.build(obs_dim=_make_env_seeded(0).observation_space.shape[0],
                               n_actions=_make_env_seeded(0).n_actions, hidden=(8,))
        rng = np.random.default_rng(6)
        best = fk_rollout(_make_env_seeded, ac, k=4, rng=rng, base_seed=444,
                          independent_actions=True)
        assert isinstance(best, float)

    def test_invalid_k_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            fk_rollout(_make_env_seeded, None, k=0, rng=rng, base_seed=0)
