"""Output heads.

A head consumes the encoder's feature vector, so any encoder can be paired with
any head and therefore with any algorithm.  Heads own the output-layer
initialisation because the right gain depends on what the output means.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import FastLinear

__all__ = ["ActorCriticHead", "DuelingQHead", "Head", "QHead"]


class Head(nn.Module):
    """Feature vector -> algorithm-specific outputs."""

    def init_outputs(self) -> None:
        """Initialise the output layers; called by the policy builder."""
        raise NotImplementedError


class ActorCriticHead(Head):
    """Policy logits and a state value, sharing one encoder."""

    def __init__(self, feature_dim: int, n_actions: int) -> None:
        super().__init__()
        self.actor = FastLinear(feature_dim, n_actions)
        self.critic = FastLinear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(features), self.critic(features).squeeze(-1)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic(features).squeeze(-1)

    def init_outputs(self) -> None:
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)


class QHead(Head):
    def __init__(self, feature_dim: int, n_actions: int) -> None:
        super().__init__()
        self.q = FastLinear(feature_dim, n_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.q(features)

    def init_outputs(self) -> None:
        nn.init.orthogonal_(self.q.weight, 0.01)
        nn.init.zeros_(self.q.bias)


class DuelingQHead(Head):
    """Q = V + A - mean(A); subtracting the mean keeps V identifiable."""

    def __init__(self, feature_dim: int, n_actions: int) -> None:
        super().__init__()
        self.value = FastLinear(feature_dim, 1)
        self.advantage = FastLinear(feature_dim, n_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def init_outputs(self) -> None:
        nn.init.orthogonal_(self.value.weight, 1.0)
        nn.init.zeros_(self.value.bias)
        nn.init.orthogonal_(self.advantage.weight, 0.01)
        nn.init.zeros_(self.advantage.bias)
