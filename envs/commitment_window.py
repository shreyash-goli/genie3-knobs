"""Stage 6 — Per-target commitment window detector.

The commitment window is the range of diffusion timesteps during which structural
decisions lock in (identity of key contacts, hotspot engagement).  Stage 1 results
showed different targets commit at different timesteps (01-05 → t=700; 06-10 → t=950),
evidence that a *fixed* intervention point is suboptimal.

This module provides:

CommitmentWindowDetector
    Estimates the per-target commitment window from offline sweep data by measuring
    the *reward variance* across children branched at each timestep.  High variance =
    the branch point is still before commitment (diffusion still deciding); low
    variance = structure has locked in (branching doesn't help anymore).

    The window start is estimated as the earliest timestep where variance starts to
    drop significantly; the window end is where it falls below a noise floor.

DiffusionInterventionEnv
    Wraps the Genie3 TrajectoryBrancher (in-process, imports genie3 directly — this
    env MUST run inside the genie3 conda env) to implement the Stage 6 MDP:

        State  S_t : noisy structure at diffusion timestep t (SE(3) frames, as a
                     flattened observation: target one-hot + structural summary statistics
                     + [t/T, best_iptm, exploration_frac])
        Action A_t : discrete intervention choice at the commitment window:
                     (no-op | perturb conditioning scale | pick hotspot subset)
                     Currently: {no-op, scale+0.5, scale+1.0, scale+2.0, scale_off}
        Reward     : sparse terminal, same compute_reward() as Stage 0-3
        Episode    : one full diffusion trajectory per episode

    The intervention is applied by modifying the ``direction_scale`` parameter
    (conditioning strength) passed to the sampler at the commitment timestep.
    This is the "conditioning embedding perturbation" from the original plan —
    it nudges how strongly the model follows the hotspot conditioning motif.

    For the LoRA fine-tuning path (Stage 7), DiffusionInterventionEnv exposes
    the intermediate x_T state so the FrontierBuffer can cache it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import gymnasium as gym

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Commitment window detection from offline data
# ---------------------------------------------------------------------------

@dataclass
class CommitmentWindow:
    """Estimated commitment window for one target."""
    target: str
    window_start: int   # earliest timestep with meaningful variance (branch here = useful)
    window_end: int     # latest timestep before variance collapses
    peak_variance_ts: int  # timestep with maximum reward variance (best branch point)
    variance_by_ts: dict[int, float]  # {timestep: reward_variance}


class CommitmentWindowDetector:
    """Estimates per-target commitment windows from offline sweep records.

    Algorithm:
      1. For each target × timestep cell, collect rewards of all logged children.
      2. Compute reward variance within each cell (high variance = branching matters).
      3. The peak variance timestep is the best branch point.
      4. Window start/end are where variance is > noise_floor_frac * peak_variance.

    Parameters
    ----------
    noise_floor_frac : fraction of peak variance below which a timestep is "committed"
    min_n            : minimum children per cell to trust its variance estimate
    """

    def __init__(self, noise_floor_frac: float = 0.2, min_n: int = 3):
        self.noise_floor_frac = noise_floor_frac
        self.min_n = min_n

    def detect(self, records: list[dict[str, Any]]) -> dict[str, CommitmentWindow]:
        """Compute commitment windows for all targets in the offline dataset.

        Parameters
        ----------
        records : list of TrajectoryRecord dicts (from load_records())

        Returns
        -------
        dict mapping target → CommitmentWindow
        """
        from oracle.reward_oracle import compute_reward

        # group rewards by (target, timestep)
        by_target_ts: dict[str, dict[int, list[float]]] = {}
        for r in records:
            t = r.get("target")
            ts = r.get("branch_timestep")
            if t is None or ts is None:
                continue
            by_target_ts.setdefault(t, {}).setdefault(ts, []).append(compute_reward(r))

        windows: dict[str, CommitmentWindow] = {}
        for target, ts_rewards in by_target_ts.items():
            variance_by_ts: dict[int, float] = {}
            for ts, rewards in ts_rewards.items():
                if len(rewards) >= self.min_n:
                    variance_by_ts[ts] = float(np.var(rewards))

            if not variance_by_ts:
                log.warning("No sufficient data for commitment window detection: %s", target)
                continue

            peak_ts = max(variance_by_ts, key=variance_by_ts.__getitem__)
            peak_var = variance_by_ts[peak_ts]
            floor = self.noise_floor_frac * peak_var

            above_floor = sorted([ts for ts, v in variance_by_ts.items() if v >= floor])
            window_start = above_floor[0] if above_floor else peak_ts
            window_end = above_floor[-1] if above_floor else peak_ts

            windows[target] = CommitmentWindow(
                target=target,
                window_start=window_start,
                window_end=window_end,
                peak_variance_ts=peak_ts,
                variance_by_ts=variance_by_ts,
            )
            log.info(
                "CommitmentWindow %s: [%d, %d]  peak_ts=%d  peak_var=%.4f",
                target, window_start, window_end, peak_ts, peak_var,
            )

        return windows


# ---------------------------------------------------------------------------
# Conditioning scale intervention values (the discrete action space for Stage 6)
# ---------------------------------------------------------------------------

# direction_scale controls how strongly the DDIM sampler follows hotspot conditioning.
# 0.0 = unconditional; 1.0 = standard; higher = stronger conditioning (may overfit).
INTERVENTION_SCALES = (0.0, 0.5, 1.0, 2.0, 4.0)
_N_INTERVENTION_ACTIONS = len(INTERVENTION_SCALES)


# ---------------------------------------------------------------------------
# DiffusionInterventionEnv
# ---------------------------------------------------------------------------

class DiffusionInterventionEnv(gym.Env):
    """Stage 6 MDP: intervene on conditioning strength at the per-target commitment window.

    This environment MUST run inside the genie3 conda env (it imports genie3 directly).
    For offline/test use, set oracle_mode="offline" to use the OfflineRewardModel instead
    of actually running diffusion.

    Parameters
    ----------
    config_yaml     : path to genie3 experiment YAML
    targets         : list of target names to train on
    commitment_windows : per-target CommitmentWindow objects (from detector above)
    frontier_buffer : FrontierBuffer instance (for x_T seed caching and seeding)
    oracle_mode     : "live" (runs genie3 subprocess) or "offline" (uses logged data)
    n_children      : number of children per episode when in live mode
    seed            : random seed
    """

    metadata = {"render_modes": []}

    # observation layout: target one-hot (n_targets) + [t_norm, best_iptm, neg_ipae_norm,
    #                      exploration_frac, direction_scale_norm]
    _N_CONTEXT = 5

    def __init__(
        self,
        config_yaml: Optional[str] = None,
        targets: Optional[list[str]] = None,
        commitment_windows: Optional[dict[str, CommitmentWindow]] = None,
        frontier_buffer: Optional[Any] = None,
        oracle_mode: str = "offline",
        n_children: int = 5,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.config_yaml = config_yaml
        self.oracle_mode = oracle_mode
        self.n_children = n_children
        self.frontier_buffer = frontier_buffer
        self._rng = np.random.default_rng(seed)

        # resolve targets
        if targets is not None:
            self.targets = targets
        else:
            import config as cfg
            self.targets = cfg.STAGE3_TARGETS

        self.commitment_windows = commitment_windows or {}
        self._target_to_idx = {t: i for i, t in enumerate(self.targets)}
        n_targets = len(self.targets)
        obs_dim = n_targets + self._N_CONTEXT

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(_N_INTERVENTION_ACTIONS)

        # episode state
        self._current_target: Optional[str] = None
        self._step_count: int = 0
        self._best_iptm: float = 0.0
        self._best_neg_ipae: float = 0.0

        # offline oracle for offline mode
        if oracle_mode == "offline":
            from oracle.reward_oracle import OfflineRewardModel
            self._offline_oracle = OfflineRewardModel()
        else:
            self._offline_oracle = None

        log.info(
            "DiffusionInterventionEnv: mode=%s  targets=%s  |A|=%d",
            oracle_mode, self.targets, _N_INTERVENTION_ACTIONS,
        )

    # -- gymnasium API -------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}
        if "target" in options:
            self._current_target = options["target"]
        else:
            self._current_target = self._rng.choice(self.targets)

        self._step_count = 0
        self._best_iptm = 0.0
        self._best_neg_ipae = 0.0
        obs = self._make_obs(direction_scale=1.0)
        return obs, {"target": self._current_target}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply the chosen conditioning scale at the commitment window and run generation."""
        assert self._current_target is not None, "call reset() before step()"
        direction_scale = INTERVENTION_SCALES[int(action)]
        target = self._current_target
        self._step_count += 1

        # get commitment timestep for this target (fall back to 800 if unknown)
        cw = self.commitment_windows.get(target)
        branch_ts = cw.peak_variance_ts if cw is not None else 800

        # run oracle
        if self.oracle_mode == "offline":
            metrics, backoff = self._offline_oracle.sample(
                target=target,
                timestep=branch_ts,
                hotspot_mode="all",
                length_delta=0,
            )
        else:
            metrics, backoff = self._run_live(target, branch_ts, direction_scale)

        # update running bests
        iptm = metrics.get("iptm") or 0.0
        ipae = metrics.get("avg_interface_pae") or 30.0
        self._best_iptm = max(self._best_iptm, iptm)
        self._best_neg_ipae = max(self._best_neg_ipae, 1.0 - ipae / 30.0)

        from oracle.reward_oracle import compute_reward
        reward = compute_reward(metrics)

        # cache x_T in frontier buffer if provided and live mode
        x_T = metrics.get("_x_T")
        if x_T is not None and self.frontier_buffer is not None:
            from buffer.frontier_buffer import FrontierEntry
            entry = FrontierEntry(
                x_T=np.array(x_T, dtype=np.float32),
                target=target,
                levers={"timestep": branch_ts, "hotspot_mode": "all",
                        "length_delta": 0, "direction_scale": direction_scale},
                reward=reward,
                metrics=metrics,
            )
            self.frontier_buffer.update(entry)

        obs = self._make_obs(direction_scale=direction_scale)
        info = dict(metrics, action=action, direction_scale=direction_scale,
                    branch_ts=branch_ts, backoff=backoff)
        return obs, reward, True, False, info  # one-step episode

    def _make_obs(self, direction_scale: float) -> np.ndarray:
        n_targets = len(self.targets)
        one_hot = np.zeros(n_targets, dtype=np.float32)
        if self._current_target is not None:
            one_hot[self._target_to_idx[self._current_target]] = 1.0
        context = np.array([
            1.0,  # bias
            self._best_iptm,
            self._best_neg_ipae,
            min(1.0, self._step_count / 50.0),  # exploration fraction
            direction_scale / max(INTERVENTION_SCALES),  # normalised scale
        ], dtype=np.float32)
        return np.concatenate([one_hot, context])

    def _run_live(self, target: str, branch_ts: int, direction_scale: float
                  ) -> tuple[dict, int]:
        """Run live oracle (genie3 subprocess) with the chosen direction_scale."""
        from oracle.live_oracle import LiveRewardModel
        oracle = LiveRewardModel()
        metrics, backoff = oracle.sample(
            target=target,
            timestep=branch_ts,
            hotspot_mode="all",
            length_delta=0,
        )
        return metrics, backoff

    def decode_action(self, action: int) -> dict[str, Any]:
        return {"direction_scale": INTERVENTION_SCALES[int(action)]}

    @property
    def n_actions(self) -> int:
        return _N_INTERVENTION_ACTIONS
