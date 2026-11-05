"""Output heads. They consume the encoder's feature vector, so any encoder can
be paired with any head and therefore with any algorithm."""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import FastLinear


class ActorCriticHead(nn.Module):
    def __init__(self, feature_dim: int, n_actions: int) -> None:
        super().__init__()
        self.actor = FastLinear(feature_dim, n_actions)
        self.critic = FastLinear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(features), self.critic(features).squeeze(-1)

    def init_outputs(self) -> None:
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)


class QHead(nn.Module):
    def __init__(self, feature_dim: int, n_actions: int) -> None:
        super().__init__()
        self.q = FastLinear(feature_dim, n_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.q(features)

    def init_outputs(self) -> None:
        nn.init.orthogonal_(self.q.weight, 0.01)
        nn.init.zeros_(self.q.bias)


class DuelingQHead(nn.Module):
    """Q = V + A - mean(A); the mean subtraction keeps V identifiable."""

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
