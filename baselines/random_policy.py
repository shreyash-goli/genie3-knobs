"""Random-action baseline -- the weak strawman (necessary lower bound, not sufficient).

A policy must beat this trivially; the *informative* bar is the contextual bandit.
All baselines expose the same minimal interface so experiments treat them uniformly:

    select(obs, target) -> action:int
    update(target, action, reward) -> None     (no-op for stateless policies)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class RandomPolicy:
    name = "random"

    def __init__(self, n_actions: int, seed: Optional[int] = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def select(self, obs: Any = None, target: Optional[str] = None) -> int:
        return int(self.rng.integers(self.n_actions))

    def update(self, target: str, action: int, reward: float) -> None:
        pass
