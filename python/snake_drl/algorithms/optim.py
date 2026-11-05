"""The one place that performs an optimizer step.

Both algorithms share the same sequence (zero gradients, backward, optional
gradient clipping, step) and the same performance rules: ``set_to_none=True``
avoids zeroing buffers, and clipping does not read the norm back to the host, so
no host synchronization is introduced.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

__all__ = ["optimizer_step"]


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[torch.nn.Parameter],
    loss: torch.Tensor,
    *,
    max_grad_norm: float = 0.0,
) -> None:
    """Backpropagate ``loss`` and apply one optimizer step.

    ``parameters`` must be re-iterable (a list, not a one-shot generator): if it
    is consumed by the clipping call, later steps silently stop clipping.  The
    algorithms store ``list(policy.parameters())`` for exactly this reason.
    """
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(list(parameters), max_grad_norm)
    optimizer.step()
