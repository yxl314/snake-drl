"""Policies: a thin wrapper turning encoder features into actions and values.

Two families exist because the algorithms need different things:

* :class:`ActorCriticPolicy` -- logits + value, for PPO;
* :class:`QPolicy` -- Q(s, .), for the DQN family.

Both apply the observation scale on the way in, so buffers can keep uint8
observations and only pay for the float conversion where it is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
from torch import distributions as dist

from ..spaces import ActionSpec, ObsSpec
from .encoders import Encoder
from .heads import ActorCriticHead, Head

__all__ = ["ActionOutput", "ActorCriticPolicy", "GreedyPolicy", "QPolicy"]


@dataclass
class ActionOutput:
    """Everything PPO needs to remember about one acting step."""

    action: torch.Tensor  # (B,) int64
    logprob: torch.Tensor  # (B,)
    value: torch.Tensor  # (B,)
    entropy: torch.Tensor  # (B,)


class GreedyPolicy(Protocol):
    """What the evaluator needs from a policy."""

    frames: int

    def greedy(self, obs: torch.Tensor) -> torch.Tensor: ...


class _PolicyBase(nn.Module):
    def __init__(self, encoder: Encoder, spec: ObsSpec) -> None:
        super().__init__()
        self.encoder = encoder
        self.obs_scale = float(spec.scale)

    @property
    def frames(self) -> int:
        return self.encoder.frames

    @property
    def feature_dim(self) -> int:
        return self.encoder.feature_dim

    def prepare(self, obs: torch.Tensor) -> torch.Tensor:
        """uint8/int -> float32 and apply ``ObsSpec.scale``.

        ``Tensor.to(float32)`` returns the input untouched when it already is
        float32, so the scaling is done out-of-place to never mutate a buffer.
        """
        x = obs.to(torch.float32)
        if self.obs_scale != 1.0:
            x = x * self.obs_scale
        return x

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.prepare(obs))


class ActorCriticPolicy(_PolicyBase):
    def __init__(
        self, encoder: Encoder, head: ActorCriticHead, spec: ObsSpec, action_spec: ActionSpec
    ) -> None:
        super().__init__(encoder, spec)
        self.head = head
        self.n_actions = action_spec.n

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value = self.head(self._features(obs))
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> ActionOutput:
        logits, value = self.forward(obs)
        distribution = dist.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return ActionOutput(
            action=action,
            logprob=distribution.log_prob(action),
            value=value,
            entropy=distribution.entropy(),
        )

    @torch.no_grad()
    def greedy(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action used by evaluation."""
        return self.act(obs, deterministic=True).action

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """State value only; skips the actor head (used for bootstrapping)."""
        return self.head.value(self._features(obs))

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-probability, entropy and value of ``actions`` under this policy."""
        logits, value = self.forward(obs)
        distribution = dist.Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), value


class QPolicy(_PolicyBase):
    def __init__(
        self, encoder: Encoder, head: Head, spec: ObsSpec, action_spec: ActionSpec
    ) -> None:
        super().__init__(encoder, spec)
        self.head = head
        self.n_actions = action_spec.n

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self._features(obs))

    @torch.no_grad()
    def act(self, obs: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        """Epsilon-greedy action; ``epsilon=0`` is the deterministic policy."""
        greedy = self.forward(obs).argmax(dim=-1)
        if epsilon <= 0.0:
            return greedy
        explore = torch.randint(0, self.n_actions, greedy.shape, device=greedy.device)
        mask = torch.rand(greedy.shape, device=greedy.device) < epsilon
        return torch.where(mask, explore, greedy)

    @torch.no_grad()
    def greedy(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action used by evaluation."""
        return self.act(obs, epsilon=0.0)
