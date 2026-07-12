"""Regression tests for policy/lora_finetune.py.

Covers two bugs found while reviewing the 2nd-iteration design (see NEXT_STEPS.md section 0):

1. train_lora_ppo used to call env.reset() then exactly one env.step() per "episode," so a
   multi-step env (DiffusionInterventionEnv) never reached terminated=True during training
   and its terminal-reward branch was unreachable.
2. RolloutBuffer.advantages_and_returns() was hardcoded for one-step episodes and would
   silently produce wrong advantages (or leak value estimates across episode boundaries) once
   multi-step episodes were buffered.

Requires torch (skipped if unavailable, e.g. outside the genie3 conda env).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from policy.lora_finetune import (  # noqa: E402
    ActorCritic,
    PPOFinetuneConfig,
    Rollout,
    RolloutBuffer,
    bias_policy_head_toward,
    train_lora_ppo,
)


class TestBiasPolicyHeadToward:
    def test_biases_initial_distribution_toward_target_action(self):
        ac = ActorCritic.build(obs_dim=4, n_actions=3, hidden=(8,))
        before = ac.policy_head.bias.detach().clone()
        bias_policy_head_toward(ac, action_idx=2, logit_bias=1.5)
        after = ac.policy_head.bias.detach()
        # only index 2 changed, by exactly +1.5
        assert float(after[2] - before[2]) == pytest.approx(1.5)
        assert float(after[0] - before[0]) == pytest.approx(0.0)
        assert float(after[1] - before[1]) == pytest.approx(0.0)

    def test_makes_target_action_more_likely_under_argmax_on_average(self):
        # a strong bias should make the biased action the argmax for most random inputs
        ac = ActorCritic.build(obs_dim=4, n_actions=3, hidden=(8,))
        bias_policy_head_toward(ac, action_idx=1, logit_bias=10.0)
        picks = []
        for _ in range(50):
            logits, _ = ac(torch.randn(1, 4))
            picks.append(int(logits.argmax(dim=-1).item()))
        assert picks.count(1) >= 40  # dominated by the biased action


class _FakeMultiStepEnv:
    """3-step episode, Discrete(2) actions, deterministic reward: 0.0 on non-terminal
    steps, 1.0 on the terminal step -- mirrors DiffusionInterventionEnv's sparse-terminal
    reward shape without needing genie3 or an oracle."""

    N_STEPS = 3

    def __init__(self):
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self._step_count = 0

    def reset(self, seed=None, options=None):
        self._step_count = 0
        return np.zeros(2, dtype=np.float32), {"target": "fake"}

    def step(self, action):
        self._step_count += 1
        terminated = self._step_count >= self.N_STEPS
        reward = 1.0 if terminated else 0.0
        obs = np.full(2, float(self._step_count), dtype=np.float32)
        return obs, reward, terminated, False, {}


class TestTrainLoraPPOMultiStepEpisodes:
    def test_episodes_reach_terminal_reward(self, tmp_path):
        env = _FakeMultiStepEnv()
        ac = ActorCritic.build(obs_dim=2, n_actions=2, hidden=(8,))
        cfg = PPOFinetuneConfig(
            total_episodes=4,
            ppo_update_freq=2,
            batch_size=4,
            n_epochs=1,
            save_every=100,
            save_dir=str(tmp_path),
        )

        log = train_lora_ppo(env, ac, cfg, verbose=False)

        assert log["total_episodes"] == 4
        assert len(log["episode_rewards"]) == 4
        # Every episode must accumulate the terminal reward of 1.0. Under the old bug
        # (env.reset() + one env.step() per "episode") this would be 0.0 every time, since
        # a fresh episode never gets past step_count==1 out of 3.
        assert all(r == pytest.approx(1.0) for r in log["episode_rewards"])

    def test_buffer_sees_full_episode_length(self, tmp_path):
        """Directly check the buffer accumulates N_STEPS rollouts per episode before an
        update, rather than exactly 1 (the old per-"episode" bug)."""
        env = _FakeMultiStepEnv()
        ac = ActorCritic.build(obs_dim=2, n_actions=2, hidden=(8,))
        seen_buffer_lengths = []

        obs, _ = env.reset()
        buffer = RolloutBuffer()
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, value = ac(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            buffer.add(Rollout(
                obs=obs, action=int(action.item()), log_prob=0.0, reward=reward,
                value=float(value.item()), done=done, metrics={},
            ))
            seen_buffer_lengths.append(len(buffer))
            obs = next_obs

        assert len(buffer) == env.N_STEPS
        assert seen_buffer_lengths[-1] == env.N_STEPS


class _FakeCommitEnv:
    """5-step episode, Discrete(3): action 2 is a "commit" that ends the episode early with
    the terminal reward, mirroring DiffusionInterventionEnv's commit action (§3.2) without
    genie3/oracle. Used to check the training loop + GAE handle an early `done`."""

    N_STEPS = 5
    COMMIT = 2

    def __init__(self):
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)
        self._step_count = 0

    def reset(self, seed=None, options=None):
        self._step_count = 0
        return np.zeros(2, dtype=np.float32), {"target": "fake"}

    def step(self, action):
        self._step_count += 1
        is_commit = int(action) == self.COMMIT
        terminated = is_commit or self._step_count >= self.N_STEPS
        reward = 1.0 if terminated else 0.0
        obs = np.full(2, float(self._step_count), dtype=np.float32)
        return obs, reward, terminated, False, {"termination_reason":
                                                 "commit" if is_commit else "timeout"}


class TestCommitActionEarlyTermination:
    def test_commit_buffers_fewer_than_n_steps(self):
        """A commit at step 2 must end the episode; the buffer sees 2 rollouts, not
        N_STEPS, and the last one carries done=True so GAE cuts the bootstrap there."""
        env = _FakeCommitEnv()
        obs, _ = env.reset()
        buffer = RolloutBuffer()
        actions = [0, env.COMMIT]  # one hotspot step then commit
        done = False
        for a in actions:
            _, value = (torch.zeros(1, 3), torch.zeros(1, 1))
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            buffer.add(Rollout(obs=obs, action=a, log_prob=0.0, reward=reward,
                               value=0.0, done=done, metrics=info))
            obs = next_obs
            if done:
                break
        assert len(buffer) == 2
        assert buffer._rollouts[-1].done is True
        assert buffer._rollouts[-1].metrics["termination_reason"] == "commit"
        # GAE runs cleanly on the truncated episode (A = R - V here since done cuts bootstrap)
        advantages, returns = buffer.advantages_and_returns()
        assert len(advantages) == 2

    def test_train_loop_handles_commit_env(self, tmp_path):
        env = _FakeCommitEnv()
        ac = ActorCritic.build(obs_dim=2, n_actions=3, hidden=(8,))
        # strongly bias toward commit so most episodes terminate at step 1
        bias_policy_head_toward(ac, action_idx=env.COMMIT, logit_bias=10.0)
        cfg = PPOFinetuneConfig(
            total_episodes=4, ppo_update_freq=2, batch_size=4, n_epochs=1,
            save_every=100, save_dir=str(tmp_path),
        )
        log = train_lora_ppo(env, ac, cfg, verbose=False)
        assert log["total_episodes"] == 4
        # committing immediately still reaches the terminal reward of 1.0
        assert all(r == pytest.approx(1.0) for r in log["episode_rewards"])


class TestRolloutBufferGAE:
    def _episode(self, rewards, values, dones):
        return [
            Rollout(obs=np.zeros(1), action=0, log_prob=0.0, reward=r, value=v, done=d,
                    metrics={})
            for r, v, d in zip(rewards, values, dones)
        ]

    def test_returns_match_advantage_plus_value(self):
        buf = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        for r in self._episode([0.0, 0.0, 1.0], [0.5, 0.5, 0.5], [False, False, True]):
            buf.add(r)
        advantages, returns = buf.advantages_and_returns()
        for adv, ret, rollout in zip(advantages, returns, buf._rollouts):
            assert ret == pytest.approx(adv + rollout.value)

    def test_no_leakage_across_episode_boundary(self):
        """Two buffers identical except for episode 2's value/reward. Episode 1's
        advantages/returns must be unaffected by what episode 2 looks like -- the `done`
        flag on episode 1's last step must cut the GAE bootstrap and backward-accumulation
        at the boundary."""
        ep1 = self._episode([0.0, 1.0], [0.5, 0.5], [False, True])

        buf_a = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        for r in ep1:
            buf_a.add(r)
        for r in self._episode([0.0], [100.0], [True]):
            buf_a.add(r)

        buf_b = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        for r in ep1:
            buf_b.add(r)
        for r in self._episode([5.0], [-50.0], [True]):
            buf_b.add(r)

        adv_a, ret_a = buf_a.advantages_and_returns()
        adv_b, ret_b = buf_b.advantages_and_returns()

        # first two (episode 1) entries must be identical regardless of episode 2's values
        assert adv_a[:2] == pytest.approx(adv_b[:2])
        assert ret_a[:2] == pytest.approx(ret_b[:2])
        # sanity: episode 2's own advantage *does* differ between the two buffers
        assert adv_a[2] != pytest.approx(adv_b[2])

    def test_one_step_episodes_collapse_to_trivial_case(self):
        """Every rollout done=True (the original one-shot direction_scale env) should
        reduce to the documented trivial case: A = R - V."""
        buf = RolloutBuffer()
        for r in [0.3, 0.7, -0.2]:
            buf.add(Rollout(obs=np.zeros(1), action=0, log_prob=0.0, reward=r, value=0.1,
                             done=True, metrics={}))
        advantages, returns = buf.advantages_and_returns()
        assert advantages == pytest.approx([0.2, 0.6, -0.3])
        assert returns == pytest.approx([0.3, 0.7, -0.2])
