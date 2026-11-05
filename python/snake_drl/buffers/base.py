"""Buffer protocols and the batch container shared by replay implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from ..registry import Registry


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    next_obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    indices: torch.Tensor  # buffer slots, needed to write back priorities
    weights: torch.Tensor  # importance-sampling weights (ones for uniform)


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

    def sample(self, batch_size: int) -> ReplayBatch: ...

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None: ...

    def __len__(self) -> int: ...


class RolloutProtocol(Protocol):
    def reset(self) -> None: ...

    def add(self, **kwargs: object) -> None: ...

    def flat(self) -> dict[str, torch.Tensor]: ...


BUFFERS: Registry[Callable[..., ReplayProtocol]] = Registry("replay buffer")
