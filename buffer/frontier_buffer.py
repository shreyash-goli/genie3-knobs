"""Frontier Buffer -- STUB ONLY (Stage 4, explicitly out of scope for this build).

Ported method *signatures* from LatProtRL (Lee et al., 2024) Algorithm 3 so the later
buffer-selection policy can be dropped in without reshaping Stage 0-3.  Nothing here is
wired into the env or the training loop, and that is intentional: building the buffer now
would prejudge two still-open design questions (see README / project spec Section "What's
Still Open"):

  * whether the buffer should be indexed jointly by the three levers
    (timestep, hotspot config, length) or by reward alone;
  * whether per-target commitment windows require a per-target adaptive index.

LatProtRL Algorithm 3 (Frontier Buffer), summarised:
  INITIALIZE(B, size)         -- fixed-capacity buffer of (state, reward, visit_count).
  TOP(B)                      -- epsilon-greedy pick of a state to resample/expand:
                                 with prob eps, the lowest-visit-count entry (explore);
                                 else softmax-over-reward sample (exploit).
  UPDATE(B, state, reward)    -- if buffer full and reward beats the current min, evict the
                                 lowest-reward entry and insert; else insert if room; bump
                                 visit_count when an existing state is re-selected.

These bodies raise NotImplementedError on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FrontierEntry:
    """One cached intermediate state.  ``state`` is intentionally untyped: in the eventual
    design it may be an x_hat_0 (Tweedie) tensor, a frozen diffusion state, or a lever-cell
    key -- that choice is part of the open Stage-4 design and must not be baked in here."""
    state: Any
    reward: float
    visit_count: int = 0
    # optional joint lever index (the "three levers" framing) -- left available, unused
    levers: Optional[dict] = field(default=None)


class FrontierBuffer:
    """Fixed-size frontier buffer (LatProtRL Algorithm 3).  STUB -- not implemented."""

    def __init__(self, size: int = 128, epsilon: float = 0.1, temperature: float = 1.0):
        self.size = size
        self.epsilon = epsilon
        self.temperature = temperature
        self.entries: list[FrontierEntry] = []

    # -- Algorithm 3 surface ----------------------------------------------------------
    @classmethod
    def initialize(cls, size: int = 128, epsilon: float = 0.1,
                   temperature: float = 1.0) -> "FrontierBuffer":
        """INITIALIZE -- create an empty fixed-capacity buffer."""
        return cls(size=size, epsilon=epsilon, temperature=temperature)

    def top(self) -> FrontierEntry:
        """TOP -- epsilon-greedy selection of an entry to resample (explore vs exploit)."""
        raise NotImplementedError("FrontierBuffer.top is a Stage-4 stub (out of scope).")

    def update(self, state: Any, reward: float, levers: Optional[dict] = None) -> None:
        """UPDATE -- insert/evict by reward; bump visit_count on re-selection."""
        raise NotImplementedError("FrontierBuffer.update is a Stage-4 stub (out of scope).")

    def __len__(self) -> int:
        return len(self.entries)
