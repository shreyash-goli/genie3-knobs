"""PPO over the discrete lever action space, via Stable-Baselines3 (do NOT hand-roll PPO).

The env is contextual-bandit-shaped (one-step episodes), so PPO here is effectively learning
a target-conditioned policy + value function over a small discrete action set.  That is fine
and intentional -- the research question is precisely whether that learned policy beats the
contextual bandit on the *same* env and interaction budget.

This module exposes:
    train_ppo(env, total_timesteps, ...) -> PPO model
    PPOPolicyWrapper(model, env)         -> .select(obs, target)/.update() shim so PPO plugs
                                            into the same evaluation harness as the baselines
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
except ImportError:  # pragma: no cover
    raise ImportError(
        "stable-baselines3 is required (pin <2.7 to keep torch 2.7.1 in the genie3 env): "
        "pip install 'stable-baselines3>=2.3,<2.7'"
    )


def train_ppo(
    env,
    total_timesteps: int = 20_000,
    seed: Optional[int] = 0,
    n_steps: int = 256,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.01,
    n_epochs: int = 10,
    verbose: int = 0,
    policy_kwargs: Optional[dict] = None,
):
    """Train a PPO policy on a GenieBranchEnv instance.

    Small MLP policy; defaults tuned for a tiny discrete action space.  ``ent_coef`` is kept
    non-trivial to keep exploration alive on the one-step MDP.  GPU is not a constraint here
    (Section 4) -- this runs fine on CPU in seconds-to-minutes.
    """
    # tiny nets on a tiny discrete env: many torch threads thrash and run *slower*.
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    monitored = Monitor(env)
    policy_kwargs = policy_kwargs or dict(net_arch=[64, 64])
    model = PPO(
        "MlpPolicy",
        monitored,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        n_epochs=n_epochs,
        gamma=0.99,
        verbose=verbose,
        policy_kwargs=policy_kwargs,
        device="cpu",
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model


class PPOPolicyWrapper:
    """Adapt a trained SB3 PPO model to the baselines' select()/update() interface so the
    evaluation harness can score it identically to fixed/random/bandit."""

    name = "ppo"

    def __init__(self, model, env, deterministic: bool = True):
        self.model = model
        self.env = env
        self.deterministic = deterministic

    def select(self, obs: Any = None, target: Optional[str] = None) -> int:
        action, _ = self.model.predict(np.asarray(obs), deterministic=self.deterministic)
        return int(action)

    def update(self, target: str, action: int, reward: float) -> None:
        pass  # PPO is trained offline-to-this-harness in train_ppo(); eval is frozen-policy
