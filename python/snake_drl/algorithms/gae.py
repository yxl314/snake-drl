"""Generalized advantage estimation, vectorized over (steps, envs)."""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(advantages, returns)`` for ``(S, N)`` rollouts.

    ``next_values[t]`` must be the value of the *true* successor of step ``t``:
    ``values[t+1]`` for a continuing transition, ``V(final_obs)`` for a truncated
    one (the caller handles that).  ``terminated`` cuts the trace; ``truncated``
    must not be in this mask, otherwise the bootstrap is silently dropped.
    """
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    nonterminal = 1.0 - terminated.to(rewards.dtype)
    for t in range(steps - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * nonterminal[t] - values[t]
        last = delta + gamma * lam * nonterminal[t] * last
        advantages[t] = last
    return advantages, advantages + values
