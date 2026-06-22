"""Stage 5 — Diffusion Frontier Buffer (LatProtRL Algorithm 3, adapted for diffusion seeds).

LatProtRL stores high-fitness latent sequence vectors z and restarts optimization from them
(Go-Explore logic).  We adapt this to diffusion: instead of z, we store the initial noise
vector x_T (shape [N_atoms, 3]) that seeded a high-reward generation.  Same idea, different
space.

Key design decisions from Stage 3 results (see experiments/RESULTS.md):
  - Buffer is indexed by (target, lever_cell), NOT reward alone.  PPO showed target-dependent
    optimal levers; a reward-only index would discard that signal.
  - Diversity is maintained via a novelty gate: new entries within
    _NOVELTY_RMSD_THRESHOLD Å of an existing entry (same target) are rejected.

Public API (extends LatProtRL Alg 3 signatures):
    buf = FrontierBuffer(size, epsilon, temperature)
    buf.initialize(targets)      # (re-)initialize per-target lists
    entry = buf.top(target)      # sample from promising region
    buf.update(entry)            # add / evict
    x_T = buf.sample_seed(...)   # used by env.reset() to seed diffusion

FrontierEntry fields:
    x_T          : np.ndarray [N_atoms, 3]  — initial noise coordinates (diffusion seed)
    target       : str                — which binder target
    levers       : dict               — {timestep, hotspot_mode, length_delta}
    reward       : float              — terminal reward from compute_reward()
    metrics      : dict               — full metrics dict (for analysis)
    visit_count  : int                — how many times this entry has been sampled
    backbone_pdb : str | None         — path to child_*.pdb if available
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# FrontierEntry
# ---------------------------------------------------------------------------

@dataclass
class FrontierEntry:
    """One entry in the frontier buffer: a promising diffusion seed + its outcome."""
    x_T: np.ndarray                      # initial noise vector [N_atoms, 3]
    target: str
    levers: dict[str, Any]               # {timestep: int, hotspot_mode: str, length_delta: int}
    reward: float
    metrics: dict[str, Any] = field(default_factory=dict)
    visit_count: int = 0
    backbone_pdb: Optional[str] = None   # path to backbone PDB for downstream analysis


# ---------------------------------------------------------------------------
# Novelty / diversity helpers
# ---------------------------------------------------------------------------

_NOVELTY_RMSD_THRESHOLD = 2.0  # Å — below this, two seeds are "too similar"


def _ca_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Cα RMSD between two coordinate arrays (no alignment — shared coordinate frame)."""
    min_len = min(len(a), len(b))
    if min_len == 0:
        return float("nan")
    diff = a[:min_len] - b[:min_len]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=-1))))


def _is_novel(x_T: np.ndarray, existing: list[FrontierEntry],
              threshold: float = _NOVELTY_RMSD_THRESHOLD) -> bool:
    """True if x_T is at least ``threshold`` Å from every existing entry."""
    for e in existing:
        if _ca_rmsd(x_T, e.x_T) < threshold:
            return False
    return True


# ---------------------------------------------------------------------------
# FrontierBuffer
# ---------------------------------------------------------------------------

class FrontierBuffer:
    """Diffusion Frontier Buffer — LatProtRL Algorithm 3 adapted for x_T seeds.

    Parameters
    ----------
    size              : max entries *per target* (evicts lowest-reward when full)
    epsilon           : Go-Explore exploitation mix: with prob epsilon sample the
                        single highest-reward entry; else softmax-weighted by reward
    temperature       : temperature for softmax reward sampling (higher = more uniform)
    novelty_threshold : minimum Cα RMSD (Å) for a new entry to pass the novelty gate
    p_frontier        : probability env.reset() draws from buffer vs fresh Gaussian noise
    """

    def __init__(
        self,
        size: int = 64,
        epsilon: float = 0.1,
        temperature: float = 1.0,
        novelty_threshold: float = _NOVELTY_RMSD_THRESHOLD,
        p_frontier: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.size = size
        self.epsilon = epsilon
        self.temperature = temperature
        self.novelty_threshold = novelty_threshold
        self.p_frontier = p_frontier
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._entries: dict[str, list[FrontierEntry]] = {}

    # -- LatProtRL Alg 3: INITIALIZE -----------------------------------------

    def initialize(
        self,
        targets: list[str],
        size: Optional[int] = None,
        epsilon: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """(Re-)initialize buffer for a set of targets, clearing all existing entries."""
        if size is not None:
            self.size = size
        if epsilon is not None:
            self.epsilon = epsilon
        if temperature is not None:
            self.temperature = temperature
        self._entries = {t: [] for t in targets}

    # -- LatProtRL Alg 3: TOP ------------------------------------------------

    def top(
        self,
        target: str,
        levers: Optional[dict[str, Any]] = None,
    ) -> Optional[FrontierEntry]:
        """Sample a promising entry for ``target`` to seed the next generation.

        Sampling:
          - With prob ``epsilon``: return the highest-reward entry (pure exploit).
          - Otherwise: softmax-weighted sample by reward over the pool.
          - If ``levers`` dict is given, restrict pool to entries with matching levers.
          - Returns None if the buffer has no entries for this target.

        Increments visit_count on the returned entry.
        """
        pool = self._entries.get(target, [])
        if levers is not None:
            pool = [e for e in pool if _levers_match(e.levers, levers)]
        if not pool:
            return None

        if self._rng.random() < self.epsilon or len(pool) == 1:
            entry = max(pool, key=lambda e: e.reward)
        else:
            rewards = np.array([e.reward for e in pool], dtype=np.float64)
            rewards = rewards - rewards.max()  # numerical stability
            weights = np.exp(rewards / self.temperature)
            weights /= weights.sum()
            idx = int(self._np_rng.choice(len(pool), p=weights))
            entry = pool[idx]

        entry.visit_count += 1
        return entry

    # -- LatProtRL Alg 3: UPDATE ---------------------------------------------

    def update(self, entry: FrontierEntry) -> bool:
        """Add an entry to the buffer.

        Acceptance rules:
          1. Novelty gate: x_T must be >= novelty_threshold Å from all existing entries
             for the same target.
          2. If buffer is full: only accept if reward > min(existing); evict the worst.

        Returns True if accepted, False if rejected.
        """
        target = entry.target
        if target not in self._entries:
            self._entries[target] = []
        existing = self._entries[target]

        if existing and not _is_novel(entry.x_T, existing, self.novelty_threshold):
            return False

        if len(existing) < self.size:
            existing.append(entry)
            return True

        worst_idx = min(range(len(existing)), key=lambda i: existing[i].reward)
        if entry.reward > existing[worst_idx].reward:
            existing[worst_idx] = entry
            return True

        return False

    # -- seed sampling (used by Stage 6 env) ---------------------------------

    def sample_seed(
        self,
        target: str,
        shape: tuple[int, ...],
        levers: Optional[dict[str, Any]] = None,
        perturb_scale: float = 0.1,
    ) -> np.ndarray:
        """Return an x_T seed for ``target``, or fresh Gaussian noise if buffer is empty.

        A small perturbation is added to the stored seed so repeated sampling from the
        same entry still explores around that region.

        Parameters
        ----------
        target        : target name
        shape         : expected shape (N_atoms, 3); used for fresh-noise fallback
        levers        : optional lever filter passed to top()
        perturb_scale : std-dev of Gaussian perturbation added to stored seed (Å)
        """
        entry = self.top(target, levers=levers)
        if entry is not None:
            return entry.x_T + self._np_rng.normal(0, perturb_scale, entry.x_T.shape).astype(np.float32)
        return self._np_rng.standard_normal(shape).astype(np.float32)

    # -- introspection -------------------------------------------------------

    def __len__(self) -> int:
        return sum(len(v) for v in self._entries.values())

    def size_for(self, target: str) -> int:
        return len(self._entries.get(target, []))

    def best_reward(self, target: str) -> float:
        pool = self._entries.get(target, [])
        return max((e.reward for e in pool), default=float("-inf"))

    def stats(self) -> dict[str, Any]:
        """Summary dict for logging/debugging."""
        out: dict[str, Any] = {}
        for t, entries in self._entries.items():
            if entries:
                rewards = [e.reward for e in entries]
                out[t] = {
                    "n": len(entries),
                    "best_reward": max(rewards),
                    "mean_reward": sum(rewards) / len(rewards),
                    "total_visits": sum(e.visit_count for e in entries),
                }
        return out


def _levers_match(a: dict, b: dict) -> bool:
    """True if all keys present in b match values in a."""
    return all(a.get(k) == v for k, v in b.items())
