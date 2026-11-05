"""Generalized advantage estimation, vectorized over (steps, envs)."""

from __future__ import annotations

import torch

__all__ = ["compute_gae"]


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor | None,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(advantages, returns)`` for ``(S, N)`` rollouts.

    ``next_values[t]`` must be the value of the *true* successor of step ``t``:
    ``values[t+1]`` for a continuing transition, ``V(final_obs)`` for a truncated
    one (the caller handles that).

    The two masks play different roles and must not be conflated:

    * ``terminated`` suppresses the bootstrap in ``delta`` (a dead end is worth
      its reward and nothing else);
    * ``terminated | truncated`` ends the trace in the recursion.  A truncated
      step still bootstraps, but the next transition belongs to a *new* episode
      (autoreset), so letting it flow into earlier advantages would import
      another episode's returns.
    """
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    nonterminal = 1.0 - terminated.to(rewards.dtype)
    cut = nonterminal if truncated is None else 1.0 - (terminated | truncated).to(rewards.dtype)
    for t in range(steps - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * nonterminal[t] - values[t]
        last = delta + gamma * lam * cut[t] * last
        advantages[t] = last
    return advantages, advantages + values
