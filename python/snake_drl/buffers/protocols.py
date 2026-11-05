"""Structural interfaces the algorithms depend on.

Algorithms only import these protocols, never a concrete buffer, which is what
keeps ``uniform``/``prioritized`` (and any user buffer) interchangeable.
"""

from __future__ import annotations

from typing import Protocol

import torch

from .batch import ReplayBatch

__all__ = ["ReplayProtocol"]


class ReplayProtocol(Protocol):
    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None: ...

    def sample(self, batch_size: int, beta: float = 1.0) -> ReplayBatch: ...

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None: ...

    def __len__(self) -> int: ...
