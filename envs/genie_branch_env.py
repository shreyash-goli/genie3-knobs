"""Gym-style environment over the three search/conditioning levers (Section 2.1).

This is an *offline* env: ``step`` does not run Genie3.  It asks the OfflineRewardModel for
a real logged child matching the chosen lever cell and returns the scalar terminal reward.
This is what makes the whole fixed/random/bandit/PPO comparison runnable with no GPU.

Episode structure (intentionally simple -- one decision per episode, a contextual-bandit-
shaped MDP):

    reset()  -> pick a target (round-robin or fixed), expose its context as the observation
    step(a)  -> apply the chosen action's levers, sample the oracle, compute terminal
                reward, ``terminated=True``.

A one-step episode is the honest shape for the MVP: the reward is sparse + terminal, and
the action is a single joint choice of (branch_timestep, length_delta, hotspot_mode).  The
class is written so a multi-step variant (sequential lever decisions, frontier-buffer
re-entry) can be added later without changing the observation/reward plumbing.

Action space is configurable:
    levers=("timestep",)                      -> Stage 1 (Discrete(n_timesteps))
    levers=("timestep","length","hotspot")    -> Stage 2/3 (Discrete(nt*nl*nh), factored)

The valid lever *values* are intersected with what the dataset actually contains for the
selected target(s), so the env never offers an action with no grounding.
"""

from __future__ import annotations

import itertools
from typing import Any, Optional, Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - gymnasium is a hard dep, this is just a clearer error
    raise ImportError("gymnasium is required: pip install 'gymnasium>=0.29'")

import config
from oracle.reward_oracle import OfflineRewardModel, RewardWeights, compute_reward


class GenieBranchEnv(gym.Env):
    """Offline env over the branch-timestep / length / hotspot levers."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        oracle: Optional[OfflineRewardModel] = None,
        targets: Optional[Sequence[str]] = None,
        levers: Sequence[str] = ("timestep",),
        reward_weights: Optional[RewardWeights] = None,
        timesteps: Optional[Sequence[int]] = None,
        length_deltas: Optional[Sequence[int]] = None,
        hotspot_modes: Optional[Sequence[str]] = None,
        fixed_target: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.oracle = oracle if oracle is not None else OfflineRewardModel()
        self.reward_weights = reward_weights or RewardWeights()
        self.levers = tuple(levers)
        for lv in self.levers:
            if lv not in ("timestep", "length", "hotspot"):
                raise ValueError(f"unknown lever {lv!r}")

        self.targets = list(targets) if targets is not None else self.oracle.targets()
        if not self.targets:
            raise ValueError("no targets available in the offline dataset")
        self.fixed_target = fixed_target
        self._target_idx = 0

        # ---- resolve valid lever values (intersect spec candidates with real data) -----
        self.timestep_values = self._resolve_lever_values(
            "timestep", timesteps, config.BRANCH_TIMESTEPS,
            lambda t: self.oracle.available_timesteps(t))
        self.length_values = self._resolve_lever_values(
            "length", length_deltas, config.LENGTH_DELTAS,
            lambda t: self.oracle.available_length_deltas(t))
        self.hotspot_values = self._resolve_lever_values(
            "hotspot", hotspot_modes, config.HOTSPOT_MODES,
            lambda t: self.oracle.available_modes(t))

        # ---- build the factored discrete action table ----------------------------------
        self._action_table = self._build_action_table()
        self.action_space = spaces.Discrete(len(self._action_table))

        # ---- observation: target one-hot + per-target context (Section 2.1) -------------
        # context = [norm_timestep_seen?, best_iptm_seen, best_neg_ipae_seen, episode_frac]
        self._n_context = 4
        obs_dim = len(self.targets) + self._n_context
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        # per-target running best (reward proxies "seen so far", part of the state)
        self._best_iptm: dict[str, float] = {t: 0.0 for t in self.targets}
        self._best_neg_ipae: dict[str, float] = {t: 0.0 for t in self.targets}
        self._history: dict[str, list[dict]] = {t: [] for t in self.targets}
        self._cur_target: Optional[str] = None

    # ---------------------------------------------------------------------------------
    # action-space construction
    # ---------------------------------------------------------------------------------
    def _resolve_lever_values(self, lever, explicit, candidates, available_fn):
        """Pick the lever values to expose.  If ``lever`` isn't active, collapse to its
        neutral default.  Otherwise use the intersection of (explicit | candidate) values
        with what is available for *every* selected target (so actions are universal)."""
        if lever not in self.levers:
            return {"timestep": [self._default_timestep()],
                    "length": [0], "hotspot": ["all"]}[lever]
        if explicit is not None:
            return list(explicit)
        wanted = set(candidates)
        common: Optional[set] = None
        for t in (self.targets if self.fixed_target is None else [self.fixed_target]):
            avail = set(available_fn(t))
            common = avail if common is None else (common & avail)
        usable = sorted((common or set()) & wanted) or sorted(common or set())
        if not usable:
            raise ValueError(f"no usable values for lever {lever!r} across targets")
        return usable

    def _default_timestep(self) -> int:
        # the timestep with the most data across selected targets (sensible neutral value)
        counts: dict[int, int] = {}
        for t in self.targets:
            for ts in self.oracle.available_timesteps(t):
                counts[ts] = counts.get(ts, 0) + 1
        return max(counts, key=counts.get) if counts else config.BRANCH_TIMESTEPS[2]

    def _build_action_table(self) -> list[dict[str, Any]]:
        table = []
        for ts, ld, hm in itertools.product(
            self.timestep_values, self.length_values, self.hotspot_values
        ):
            table.append({"timestep": ts, "length_delta": ld, "hotspot_mode": hm})
        return table

    def decode_action(self, action: int) -> dict[str, Any]:
        """Map a discrete action index to its lever dict (public: used by experiments)."""
        return dict(self._action_table[int(action)])

    # ---------------------------------------------------------------------------------
    # observation
    # ---------------------------------------------------------------------------------
    def _make_obs(self, target: str) -> np.ndarray:
        onehot = np.zeros(len(self.targets), dtype=np.float32)
        onehot[self.targets.index(target)] = 1.0
        ctx = np.array([
            1.0,  # placeholder: "a decision is pending" flag (room for timestep-in-episode)
            self._best_iptm[target],
            self._best_neg_ipae[target],
            min(1.0, len(self._history[target]) / 64.0),  # how much explored this round
        ], dtype=np.float32)
        return np.concatenate([onehot, ctx]).astype(np.float32)

    # ---------------------------------------------------------------------------------
    # gym API
    # ---------------------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.fixed_target is not None:
            target = self.fixed_target
        elif options and "target" in options:
            target = options["target"]
        else:
            target = self.targets[self._target_idx % len(self.targets)]
            self._target_idx += 1
        self._cur_target = target
        return self._make_obs(target), {"target": target}

    def step(self, action: int):
        target = self._cur_target
        if target is None:
            raise RuntimeError("call reset() before step()")
        levers = self.decode_action(action)

        metrics, backoff = self.oracle.sample(
            target=target,
            timestep=levers["timestep"],
            hotspot_mode=levers["hotspot_mode"],
            length_delta=levers["length_delta"],
        )
        reward = compute_reward(metrics, history=self._history[target],
                                weights=self.reward_weights)

        # update "seen so far" proxies (part of the observable state for the next episode)
        if metrics.get("iptm") is not None:
            self._best_iptm[target] = max(self._best_iptm[target], float(metrics["iptm"]))
        if metrics.get("avg_interface_pae") is not None:
            neg = max(0.0, 1.0 - metrics["avg_interface_pae"] / self.reward_weights.ipae_scale)
            self._best_neg_ipae[target] = max(self._best_neg_ipae[target], neg)
        self._history[target].append(metrics)

        info = {
            "target": target,
            "action": levers,
            "backoff": OfflineRewardModel.BACKOFF_LABELS[backoff],
            "backoff_level": backoff,
            "iptm": metrics.get("iptm"),
            "avg_interface_pae": metrics.get("avg_interface_pae"),
            "complex_success": metrics.get("complex_success"),
            "hotspot_coverage": metrics.get("hotspot_coverage"),
            "child_id": metrics.get("child_id"),
        }
        terminated, truncated = True, False  # one-step (sparse terminal reward) episode
        obs = self._make_obs(target)
        return obs, float(reward), terminated, truncated, info

    def reset_round(self) -> None:
        """Clear per-target 'seen so far' history (start a fresh design round)."""
        self._best_iptm = {t: 0.0 for t in self.targets}
        self._best_neg_ipae = {t: 0.0 for t in self.targets}
        self._history = {t: [] for t in self.targets}
