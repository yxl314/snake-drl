"""Replay and rollout buffers."""

from __future__ import annotations

from typing import Optional

import torch

from ..config import BufferConfig
from ..spaces import ObsSpec
from .base import BUFFERS, ReplayBatch, ReplayProtocol, RolloutProtocol
from .replay import PrioritizedReplayBuffer, UniformReplayBuffer
from .rollout import RolloutBuffer
from .sumtree import SumTree

__all__ = [
    "BUFFERS",
    "PrioritizedReplayBuffer",
    "ReplayBatch",
    "ReplayProtocol",
    "RolloutBuffer",
    "RolloutProtocol",
    "SumTree",
    "UniformReplayBuffer",
    "build_replay",
]


def build_replay(
    cfg: BufferConfig,
    obs_spec: ObsSpec,
    frames: int,
    device: torch.device | str,
    *,
    pin_memory: bool = False,
    seed: int = 0,
) -> ReplayProtocol:
    factory = BUFFERS.get(cfg.type)
    return factory(
        cfg=cfg, obs_spec=obs_spec, frames=frames, device=device, pin_memory=pin_memory, seed=seed
    )
