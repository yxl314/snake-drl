"""Replay and rollout buffers.

Importing this package registers the built-in replay buffers; ``build_replay``
turns a :class:`~snake_drl.config.BufferConfig` into one of them.
"""

from __future__ import annotations

import torch

from ..config import BufferConfig
from ..spaces import ObsSpec
from .batch import ReplayBatch
from .protocols import ReplayProtocol
from .registry import BUFFERS
from .replay import PrioritizedReplayBuffer, UniformReplayBuffer  # noqa: F401 (registers)
from .rollout import RolloutBuffer
from .sumtree import SumTree

__all__ = [
    "BUFFERS",
    "PrioritizedReplayBuffer",
    "ReplayBatch",
    "ReplayProtocol",
    "RolloutBuffer",
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
    """Instantiate the replay buffer named by ``cfg.type``."""
    if cfg.type not in BUFFERS:
        available = ", ".join(BUFFERS.names())
        raise ValueError(f"unknown replay buffer type {cfg.type!r}; available: {available}")
    return BUFFERS.create(
        cfg.type,
        cfg=cfg,
        obs_spec=obs_spec,
        frames=frames,
        device=device,
        pin_memory=pin_memory,
        seed=seed,
    )
