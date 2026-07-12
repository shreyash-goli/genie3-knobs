"""Contextual-bandit baseline -- THE baseline that determines whether PPO earns its keep.

The MDP here is contextual-bandit-shaped: each episode is a single decision conditioned on
the target (the context), with a sparse terminal reward.  A per-target action-value table
with a principled exploration rule is therefore the *correct* non-RL learner for this
problem, not a strawman.  If PPO cannot beat this, the "RL" framing is not buying anything
on this action space (Stage-1 stopping rule, Section 2.2).

Design choices made deliberately strong (per the spec: "build this with real care"):

* **Contextual**: separate action-value estimates per target.  The context is the target
  identity, recovered from the observation one-hot (so the bandit sees exactly what PPO's
  policy network sees -- a fair comparison).
* **Exploration**: UCB1 (default) or epsilon-greedy, both standard and well-understood.
* **Online**: incremental sample-mean updates from the same env interactions PPO trains on,
  under the same interaction budget -> apples-to-apples.
* **Optional warm-start**: can be primed from the offline dataset's per-cell means (the
  "lookup table you already have"), to test the strongest possible non-RL bar.

Interface: select(obs, target) -> action; update(target, action, reward) -> None.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from oracle.reward_oracle import OfflineRewardModel, RewardWeights, compute_reward


class ContextualBandit:
    name = "bandit"

    def __init__(
        self,
        env,
        exploration: str = "ucb",      # "ucb" | "epsilon"
        c: float = 2.0,                # UCB exploration coefficient
        epsilon: float = 0.1,          # epsilon-greedy rate
        warm_start: bool = False,      # prime estimates from offline dataset means
        warm_start_pseudocount: int = 3,
        oracle: Optional[OfflineRewardModel] = None,
        weights: Optional[RewardWeights] = None,
        seed: Optional[int] = None,
        n_actions: Optional[int] = None,
    ):
        self.env = env
        self.exploration = exploration
        self.c = c
        self.epsilon = epsilon
        # n_actions defaults to the full action space, but can be restricted to a prefix of the
        # arms -- e.g. the windowed DiffusionInterventionEnv exposes Discrete(4) where index 3 is
        # the commit action; passing n_actions=len(HOTSPOT_MODES) makes the bandit the exact
        # context-conditioned analogue of the fixed/random hotspot-only baselines.
        self.n_actions = n_actions if n_actions is not None else env.action_space.n
        self.targets = list(env.targets)
        self.rng = np.random.default_rng(seed)
        self.weights = weights or env.reward_weights
        self.oracle = oracle or env.oracle

        # per-target action-value tables
        self._q: dict[str, np.ndarray] = {
            t: np.zeros(self.n_actions, dtype=np.float64) for t in self.targets
        }
        self._n: dict[str, np.ndarray] = {
            t: np.zeros(self.n_actions, dtype=np.float64) for t in self.targets
        }
        self._t: dict[str, int] = {t: 0 for t in self.targets}

        if warm_start:
            self._warm_start(warm_start_pseudocount)

    # ---------------------------------------------------------------------------------
    def _warm_start(self, pseudocount: int) -> None:
        """Prime per-(target,action) estimates from offline-dataset cell means.

        Each populated cell seeds its action's value with ``pseudocount`` virtual pulls at
        the cell's empirical mean reward -- i.e. the lookup table the spec warns PPO must
        beat.  Cells the dataset doesn't cover are left cold (the bandit must explore them).
        """
        for target in self.targets:
            for action in range(self.n_actions):
                levers = self.env.decode_action(action)
                pool = self.oracle.by_cell.get(
                    (target, levers["timestep"], levers["hotspot_mode"],
                     levers["length_delta"]), [])
                if not pool:
                    continue
                mean_r = float(np.mean([compute_reward(r, weights=self.weights)
                                        for r in pool]))
                self._q[target][action] = mean_r
                self._n[target][action] = pseudocount
                self._t[target] += pseudocount

    # ---------------------------------------------------------------------------------
    def _target_of(self, obs: Any, target: Optional[str]) -> str:
        if target is not None:
            return target
        # recover target from the observation one-hot (first len(targets) dims)
        idx = int(np.argmax(np.asarray(obs)[: len(self.targets)]))
        return self.targets[idx]

    def select(self, obs: Any = None, target: Optional[str] = None) -> int:
        target = self._target_of(obs, target)
        q, n = self._q[target], self._n[target]

        # always try each action once before exploiting
        unseen = np.where(n == 0)[0]
        if len(unseen) > 0:
            return int(self.rng.choice(unseen))

        if self.exploration == "epsilon":
            if self.rng.random() < self.epsilon:
                return int(self.rng.integers(self.n_actions))
            return int(self._argmax_random_tie(q))

        # UCB1
        total = self._t[target]
        bonus = self.c * np.sqrt(np.log(max(total, 1)) / n)
        return int(self._argmax_random_tie(q + bonus))

    def _argmax_random_tie(self, values: np.ndarray) -> int:
        m = values.max()
        cands = np.where(values >= m - 1e-12)[0]
        return int(self.rng.choice(cands))

    def update(self, target: str, action: int, reward: float) -> None:
        self._n[target][action] += 1
        self._t[target] += 1
        n = self._n[target][action]
        # incremental sample mean
        self._q[target][action] += (reward - self._q[target][action]) / n

    # ---------------------------------------------------------------------------------
    def greedy_action(self, target: str) -> int:
        """The action the bandit would commit to for a target (for analysis/Stage 3)."""
        return int(self._argmax_random_tie(self._q[target]))

    def policy_table(self) -> dict[str, int]:
        """Per-target greedy action -- used by Stage 3 to inspect target-dependent behaviour."""
        return {t: self.greedy_action(t) for t in self.targets}
