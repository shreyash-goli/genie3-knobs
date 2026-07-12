"""Windowed MDP — per-step hotspot selection across the diffusion commitment window.

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

DiffusionInterventionEnv
    A 10-step MDP over the commitment window.  At each step the policy picks a
    hotspot mode; the episode terminates after N_WINDOW_STEPS steps and emits a
    single sparse terminal reward.

    State  S_t : target one-hot + [step_progress, t_norm, length_delta_norm,
                 best_iptm, best_neg_ipae]
    Action A_t : Discrete(3) — index into HOTSPOT_MODES
    Reward     : 0 for steps 0..N-2; compute_reward(metrics) at step N-1 (terminal)
    Length     : sampled once at reset() from LENGTH_DELTAS; fixed for the episode

    In offline mode each step is a cheap oracle lookup (no GPU); in live mode each
    step maps to one branch point within the commitment window, not one full oracle
    call per step — the full ColabFold eval only runs at the final step.

Design rationale:
    Making hotspot mode a per-step decision over 10 diffusion timesteps gives PPO a
    genuine short-horizon credit-assignment problem to exploit.  Without this the env
    is a one-shot bandit and PPO adds overhead without signal.
    Stopping rule: if sparse-terminal proves too sample-inefficient (no learning
    signal in the first 200 episodes), add iCS as a per-step intermediate reward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import gymnasium as gym

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action / lever constants
# ---------------------------------------------------------------------------

HOTSPOT_MODES: tuple[str, ...] = ("all", "ablate_competitors", "missed_only")
LENGTH_DELTAS: tuple[int, ...] = (0, 60)
N_WINDOW_STEPS: int = 10

# Learned "commit" action (NEXT_STEPS.md §3.2): the policy can end the episode early and
# take the terminal reward from the current step instead of always running the full window.
# It is *not* a member of HOTSPOT_MODES (those are the 3 conditioning arms the fixed-bandit
# baselines enumerate) -- it is an extra action index appended after them.
COMMIT_ACTION: int = len(HOTSPOT_MODES)

# Default commitment window bounds used when per-target data is unavailable.
_DEFAULT_WINDOW_START: int = 700
_DEFAULT_WINDOW_END: int = 950


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
        """Compute commitment windows for all targets in the offline dataset."""
        from oracle.reward_oracle import compute_reward

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
# DiffusionInterventionEnv — windowed MDP
# ---------------------------------------------------------------------------

def _timestep_schedule(window_start: int, window_end: int, n: int) -> list[int]:
    """Return n uniformly-spaced integer timesteps in [window_start, window_end]."""
    if n == 1:
        return [window_start]
    step = (window_end - window_start) / (n - 1)
    return [int(round(window_start + i * step)) for i in range(n)]


class DiffusionInterventionEnv(gym.Env):
    """Windowed MDP: 10 per-step hotspot decisions across the diffusion commitment window.

    Action space  : Discrete(3) — index into HOTSPOT_MODES
    Observation   : target one-hot + [step_progress, t_norm, length_delta_norm,
                    best_iptm, best_neg_ipae]  (all float32)
    Reward        : intermediate_reward_scale * compute_reward(metrics) at steps 0..N-2;
                    compute_reward(metrics) at step N-1 (terminal, unscaled)

    intermediate_reward_scale=0.0 reproduces the original sparse-terminal behaviour.
    intermediate_reward_scale=0.1 gives a small per-step shaping signal (iCS proxy)
    while keeping the terminal reward the dominant signal.

    window_start_override / window_end_override bypass per-target CommitmentWindow
    detection with a fixed boundary, used for the window-placement sweep.

    Parameters
    ----------
    targets                  : list of target names to train on
    commitment_windows        : per-target CommitmentWindow (from CommitmentWindowDetector)
    frontier_buffer           : FrontierBuffer instance for x_T seed caching (optional)
    oracle_mode               : "offline" (uses OfflineRewardModel) or "live" (runs genie3)
    n_children                : children per oracle call when oracle_mode="live"
    intermediate_reward_scale : scale factor for per-step intermediate reward (default 0.0)
    window_start_override     : override window start for all targets (sweep use)
    window_end_override       : override window end for all targets (sweep use)
    seed                      : random seed
    """

    metadata = {"render_modes": []}

    # step_progress, t_norm, length_delta_norm, best_iptm, best_neg_ipae,
    # + per-hotspot-mode usage counts (action history, §3.2/§3.3)
    _N_CONTEXT = 5 + len(HOTSPOT_MODES)

    def __init__(
        self,
        targets: Optional[list[str]] = None,
        commitment_windows: Optional[dict[str, CommitmentWindow]] = None,
        frontier_buffer: Optional[Any] = None,
        oracle_mode: str = "offline",
        n_children: int = 5,
        intermediate_reward_scale: float = 0.0,
        window_start_override: Optional[int] = None,
        window_end_override: Optional[int] = None,
        mask_target_onehot: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.oracle_mode = oracle_mode
        self.n_children = n_children
        self.frontier_buffer = frontier_buffer
        self.intermediate_reward_scale = intermediate_reward_scale
        self.window_start_override = window_start_override
        self.window_end_override = window_end_override
        # When True the target one-hot block stays all-zeros, so the policy cannot tell
        # which target it is conditioning. Used by the §3.1 ablation to test whether the
        # windowed-MDP PPO win is cross-target specialization (needs the one-hot) vs.
        # genuine within-episode sequential learning (would survive without it). Keeps
        # obs_dim identical so no other code changes are needed.
        self.mask_target_onehot = mask_target_onehot
        self._rng = np.random.default_rng(seed)

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
        # +1 for the commit action (§3.2).
        self.action_space = gym.spaces.Discrete(len(HOTSPOT_MODES) + 1)

        # episode state
        self._current_target: Optional[str] = None
        self._current_length_delta: int = 0
        self._timestep_sched: list[int] = []
        self._step_count: int = 0
        self._best_iptm: float = 0.0
        self._best_neg_ipae: float = 0.0
        self._action_counts = np.zeros(len(HOTSPOT_MODES), dtype=np.float32)
        self._last_action: Optional[int] = None

        if oracle_mode == "offline":
            import random as _random
            from oracle.reward_oracle import OfflineRewardModel
            # Seed the oracle's child-sampling RNG when the env is seeded, so a fixed
            # (config, seed) gives reproducible rewards. Without this the oracle draws a
            # random logged child per cell each run, adding a large (~±0.024) noise floor
            # that swamps the small PPO-vs-fixed margins (see §6 findings in NEXT_STEPS.md).
            oracle_rng = _random.Random(seed) if seed is not None else None
            self._oracle = OfflineRewardModel(rng=oracle_rng)
        else:
            self._oracle = None  # live oracle constructed lazily per-call

        log.info(
            "DiffusionInterventionEnv: mode=%s  targets=%s  |A|=%d  steps=%d",
            oracle_mode, self.targets, len(HOTSPOT_MODES) + 1, N_WINDOW_STEPS,
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
        self._current_target = options.get("target") or str(
            self._rng.choice(self.targets)
        )
        self._current_length_delta = int(
            options.get("length_delta", self._rng.choice(LENGTH_DELTAS))
        )

        cw = self.commitment_windows.get(self._current_target)
        ws = self.window_start_override if self.window_start_override is not None else (
            cw.window_start if cw is not None else _DEFAULT_WINDOW_START)
        we = self.window_end_override if self.window_end_override is not None else (
            cw.window_end if cw is not None else _DEFAULT_WINDOW_END)
        self._timestep_sched = _timestep_schedule(ws, we, N_WINDOW_STEPS)

        self._step_count = 0
        self._best_iptm = 0.0
        self._best_neg_ipae = 0.0
        self._action_counts = np.zeros(len(HOTSPOT_MODES), dtype=np.float32)
        self._last_action = None

        obs = self._make_obs()
        return obs, {
            "target": self._current_target,
            "length_delta": self._current_length_delta,
            "timestep_schedule": self._timestep_sched,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._current_target is not None, "call reset() before step()"
        assert 0 <= int(action) <= len(HOTSPOT_MODES), f"invalid action {action}"

        action = int(action)
        is_commit = action == COMMIT_ACTION
        # A commit re-applies the last conditioning mode at the current timestep and ends the
        # episode; on the very first step (no prior mode) it falls back to HOTSPOT_MODES[0].
        if is_commit:
            hotspot_mode = HOTSPOT_MODES[self._last_action] if self._last_action is not None \
                else HOTSPOT_MODES[0]
        else:
            hotspot_mode = HOTSPOT_MODES[action]
            self._action_counts[action] += 1.0
            self._last_action = action
        timestep = self._timestep_sched[self._step_count]
        target = self._current_target
        length_delta = self._current_length_delta
        # A commit forces terminal handling: the full oracle must fire (matching the one-call-
        # per-episode invariant in live mode), and the episode ends now.
        is_terminal_step = is_commit or (self._step_count + 1) >= N_WINDOW_STEPS

        metrics, backoff = self._query_oracle(
            target, timestep, hotspot_mode, length_delta, is_terminal_step
        )

        iptm = metrics.get("iptm") or 0.0
        ipae = metrics.get("avg_interface_pae") or 30.0
        self._best_iptm = max(self._best_iptm, iptm)
        self._best_neg_ipae = max(self._best_neg_ipae, 1.0 - ipae / 30.0)

        self._step_count += 1
        terminated = is_commit or self._step_count >= N_WINDOW_STEPS
        # "commit" when the policy chose to end early, else "timeout" (window elapsed).
        # (Invalid-config / no-op reasons from §3.2 are not reachable with today's discrete
        # lever space -- they become relevant only if continuous levers are added later.)
        termination_reason = "commit" if is_commit else ("timeout" if terminated else None)

        from oracle.reward_oracle import compute_reward
        if terminated:
            reward = float(compute_reward(metrics))
            self._maybe_cache_x_T(metrics, target, timestep, hotspot_mode,
                                  length_delta, reward)
        else:
            reward = self.intermediate_reward_scale * float(compute_reward(metrics))

        obs = self._make_obs()
        info = dict(
            metrics,
            action=action,
            hotspot_mode=hotspot_mode,
            timestep=timestep,
            length_delta=length_delta,
            backoff=backoff,
            step=self._step_count,
            commit=is_commit,
            termination_reason=termination_reason,
        )
        return obs, reward, terminated, False, info

    # -- internal helpers ----------------------------------------------------

    def _query_oracle(
        self, target: str, timestep: int, hotspot_mode: str, length_delta: int,
        is_terminal_step: bool = True,
    ) -> tuple[dict[str, Any], int]:
        if self.oracle_mode == "offline":
            return self._oracle.sample(
                target=target,
                timestep=timestep,
                hotspot_mode=hotspot_mode,
                length_delta=length_delta,
            )
        # Live mode: the oracle is expensive (~10 min: full Genie3 -> ProteinMPNN ->
        # ColabFold). Pay that cost once per episode, at the terminal step only -- matching
        # the windowed MDP's documented design ("10 branch points, not 10 full oracle
        # calls"). Non-terminal steps get an empty metrics dict (compute_reward() -> 0.0)
        # until a genuinely cheap intermediate signal exists (NEXT_STEPS.md section 7.1); calling
        # the live oracle at every one of the 10 steps made a live episode cost ~100 min
        # instead of ~10 min, silently blowing the Stage 7 SLURM time budget.
        if not is_terminal_step:
            return {}, 0
        from oracle.live_oracle import LiveRewardModel
        oracle = LiveRewardModel()
        return oracle.sample(
            target=target,
            timestep=timestep,
            hotspot_mode=hotspot_mode,
            length_delta=length_delta,
        )

    def _make_obs(self) -> np.ndarray:
        n_targets = len(self.targets)
        one_hot = np.zeros(n_targets, dtype=np.float32)
        if self._current_target is not None and not self.mask_target_onehot:
            one_hot[self._target_to_idx[self._current_target]] = 1.0

        step_progress = self._step_count / N_WINDOW_STEPS
        if self._timestep_sched:
            t_norm = self._timestep_sched[min(self._step_count, N_WINDOW_STEPS - 1)] / 1000.0
        else:
            t_norm = 0.0
        length_delta_norm = float(self._current_length_delta) / max(LENGTH_DELTAS)

        context = np.array([
            step_progress,
            t_norm,
            length_delta_norm,
            self._best_iptm,
            self._best_neg_ipae,
        ], dtype=np.float32)
        # Action history: per-mode usage counts so far, normalized by the window length.
        # Order-invariant memory of which conditioning modes have been tried this episode
        # (§3.2/§3.3). Appended AFTER the 5 scalar context features so existing obs indices
        # (one-hot block, step_progress at index n_targets) are unchanged.
        action_history = self._action_counts / float(N_WINDOW_STEPS)
        return np.concatenate([one_hot, context, action_history])

    def _maybe_cache_x_T(
        self,
        metrics: dict[str, Any],
        target: str,
        timestep: int,
        hotspot_mode: str,
        length_delta: int,
        reward: float,
    ) -> None:
        x_T = metrics.get("_x_T") or metrics.get("x_T")
        if x_T is not None and self.frontier_buffer is not None:
            try:
                from buffer.frontier_buffer import FrontierEntry
                entry = FrontierEntry(
                    x_T=np.array(x_T, dtype=np.float32),
                    target=target,
                    levers={"timestep": timestep, "hotspot_mode": hotspot_mode,
                            "length_delta": length_delta},
                    reward=reward,
                    metrics=metrics,
                )
                self.frontier_buffer.update(entry)
            except Exception as e:
                log.warning("FrontierBuffer update failed: %s", e)

    def decode_action(self, action: int) -> dict[str, Any]:
        if int(action) == COMMIT_ACTION:
            return {"commit": True}
        return {"hotspot_mode": HOTSPOT_MODES[int(action)]}

    @property
    def n_actions(self) -> int:
        return len(HOTSPOT_MODES) + 1
