"""Policy assembly from config. This is the only place that maps a config to a
concrete network, which keeps the algorithms free of ``if encoder == ...``."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..config import NetworkConfig
from ..spaces import ActionSpec, ObsSpec
from .encoders import Encoder, build_encoder
from .heads import ActorCriticHead, DuelingQHead, QHead
from .policies import ActorCriticPolicy, QPolicy

FAMILIES = ("actor_critic", "q")


def _orthogonal_(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def build_policy(
    cfg: NetworkConfig,
    obs_spec: ObsSpec,
    action_spec: ActionSpec,
    *,
    family: str,
    device: torch.device | str = "cpu",
    dueling: bool = False,
) -> nn.Module:
    """Build the policy for ``family`` ("actor_critic" for PPO, "q" for DQN)."""
    if family not in FAMILIES:
        raise ValueError(f"policy family must be one of {FAMILIES}, got {family!r}")
    encoder: Encoder = build_encoder(obs_spec, cfg.encoder)
    feature_dim = encoder.feature_dim

    if cfg.head != "auto" and cfg.head == "actor_critic" and family != "actor_critic":
        raise ValueError("network.head=actor_critic is only valid for PPO")
    if cfg.head in ("q", "dueling_q") and family != "q":
        raise ValueError(f"network.head={cfg.head} is only valid for the DQN family")

    if family == "actor_critic":
        head: nn.Module = ActorCriticHead(feature_dim, action_spec.n)
        policy: nn.Module = ActorCriticPolicy(encoder, head, obs_spec, action_spec)
    else:
        head = DuelingQHead(feature_dim, action_spec.n) if dueling else QHead(feature_dim, action_spec.n)
        policy = QPolicy(encoder, head, obs_spec, action_spec)

    if cfg.ortho_init:
        _orthogonal_(encoder)
        _orthogonal_(head)
        head.init_outputs()  # type: ignore[attr-defined]
    return policy.to(torch.device(device))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
