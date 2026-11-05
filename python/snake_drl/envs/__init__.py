"""Environments and their registry.

Every environment is a batch of independent games with the same interface, and
every factory takes ``(cfg, *, seed=None)``.  Importing this package registers
the built-in environments; selecting one is a config lookup::

    make_env(EnvConfig(type="snake", num_envs=256))
"""

from __future__ import annotations

from ..config import EnvConfig
from .base import StepResult, VectorEnv
from .registry import ENVS
from .snake import SnakeVecEnv  # noqa: F401  (registers "snake")
from .toy import BanditEnv, CorridorEnv  # noqa: F401  (registers "bandit"/"corridor")

__all__ = [
    "ENVS",
    "BanditEnv",
    "CorridorEnv",
    "SnakeVecEnv",
    "StepResult",
    "VectorEnv",
    "make_env",
]


def make_env(cfg: EnvConfig, *, seed: int | None = None) -> VectorEnv:
    """Instantiate the environment named by ``cfg.type``."""
    if cfg.type not in ENVS:
        available = ", ".join(ENVS.names())
        raise ValueError(f"unknown environment type {cfg.type!r}; available: {available}")
    return ENVS.create(cfg.type, cfg=cfg, seed=seed)
