"""Policy assembly from config.

This is the only place that maps a config to a concrete network, which keeps the
algorithms free of ``if encoder == ...`` branching.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import NetworkConfig
from ..spaces import ActionSpec, ObsSpec
from .encoders import build_encoder
from .heads import ActorCriticHead, DuelingQHead, Head, QHead
from .policies import ActorCriticPolicy, QPolicy

__all__ = ["build_policy", "count_parameters"]

_FAMILIES = ("actor_critic", "q")


def _orthogonal_(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(layer.weight, gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def build_policy(
    cfg: NetworkConfig,
    obs_spec: ObsSpec,
    action_spec: ActionSpec,
    *,
    family: str,
    device: torch.device | str = "cpu",
    dueling: bool = False,
) -> ActorCriticPolicy | QPolicy:
    """Build the policy for ``family`` ("actor_critic" for PPO, "q" for DQN)."""
    if family not in _FAMILIES:
        raise ValueError(f"policy family must be one of {_FAMILIES}, got {family!r}")
    _check_head(cfg.head, family, dueling)

    encoder = build_encoder(obs_spec, cfg.encoder)
    head: Head
    if family == "actor_critic":
        head = ActorCriticHead(encoder.feature_dim, action_spec.n)
        policy: ActorCriticPolicy | QPolicy = ActorCriticPolicy(
            encoder, head, obs_spec, action_spec
        )
    else:
        head = (
            DuelingQHead(encoder.feature_dim, action_spec.n)
            if dueling
            else QHead(encoder.feature_dim, action_spec.n)
        )
        policy = QPolicy(encoder, head, obs_spec, action_spec)

    if cfg.ortho_init:
        _orthogonal_(encoder)
        _orthogonal_(head)
        head.init_outputs()
    return policy.to(torch.device(device))


def _check_head(head: str, family: str, dueling: bool) -> None:
    """Reject a head that disagrees with the algorithm instead of ignoring it."""
    if head == "actor_critic" and family != "actor_critic":
        raise ValueError("network.head=actor_critic is only valid for PPO")
    if head in ("q", "dueling_q") and family != "q":
        raise ValueError(f"network.head={head} is only valid for the DQN family")
    if head == "dueling_q" and not dueling:
        raise ValueError(
            "network.head=dueling_q needs a dueling algorithm: set algorithm.type=d3qn "
            "or algorithm.dueling=true"
        )
    if head == "q" and dueling:
        raise ValueError(
            "network.head=q contradicts a dueling algorithm; use head=auto (or dueling_q)"
        )


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
