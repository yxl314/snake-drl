"""The minibatch returned by replay sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["ReplayBatch"]


@dataclass
class ReplayBatch:
    """A sampled minibatch, already gathered into pre-allocated buffers.

    ``to()`` exists because the buffer may live on a different device than the
    network (CPU storage + GPU compute, for example); it is a no-op when they
    match.
    """

    obs: torch.Tensor
    next_obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    indices: torch.Tensor  # buffer slots, needed to write back priorities
    weights: torch.Tensor  # importance-sampling weights (ones for uniform)

    def to(self, device: torch.device, *, non_blocking: bool = False) -> ReplayBatch:
        if self.obs.device == device:
            return self
        move = lambda t: t.to(device, non_blocking=non_blocking)  # noqa: E731
        return ReplayBatch(
            obs=move(self.obs),
            next_obs=move(self.next_obs),
            action=move(self.action),
            reward=move(self.reward),
            terminated=move(self.terminated),
            truncated=move(self.truncated),
            indices=move(self.indices),
            weights=move(self.weights),
        )
