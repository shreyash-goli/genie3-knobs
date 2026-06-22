"""Fixed-heuristic baseline -- "always use the single best-known action".

This is the "one fixed heuristic for everyone" policy that a learned, target-conditioned
policy must improve on.  It picks ONE action (the same for every target) by reading the
offline dataset's marginal: the action with the highest mean reward pooled across all
selected targets.  It does NOT adapt per target -- that non-adaptiveness is the point
(Stage 3 asks whether a learned policy can beat a single global knob setting).

Interface matches the other baselines: select(obs, target) -> action, update() is a no-op.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional, Sequence

from oracle.reward_oracle import OfflineRewardModel, RewardWeights, compute_reward


class FixedHeuristic:
    name = "fixed"

    def __init__(self, env, oracle: Optional[OfflineRewardModel] = None,
                 weights: Optional[RewardWeights] = None):
        self.env = env
        self.oracle = oracle or env.oracle
        self.weights = weights or env.reward_weights
        self.best_action = self._fit()

    def _fit(self) -> int:
        """Choose the single action index with the best pooled mean reward over targets."""
        sums: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        target_set = set(self.env.targets)
        # index records once by (target, timestep, mode, length) via the oracle cells
        for action in range(self.env.action_space.n):
            levers = self.env.decode_action(action)
            for target in self.env.targets:
                pool = self.oracle.by_cell.get(
                    (target, levers["timestep"], levers["hotspot_mode"],
                     levers["length_delta"]), [])
                for rec in pool:
                    sums[action] += compute_reward(rec, weights=self.weights)
                    counts[action] += 1
        means = {a: sums[a] / counts[a] for a in counts if counts[a] > 0}
        if not means:
            return 0
        return max(means, key=means.get)

    def select(self, obs: Any = None, target: Optional[str] = None) -> int:
        return self.best_action

    def update(self, target: str, action: int, reward: float) -> None:
        pass
