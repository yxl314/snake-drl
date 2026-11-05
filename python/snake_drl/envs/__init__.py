"""Environments and the environment registry."""

from __future__ import annotations

from typing import Optional

from ..config import EnvConfig
from .base import ENVS, StepResult, VectorEnv
from .snake import SnakeVecEnv, make_snake  # noqa: F401  (registers "snake")
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


def make_env(cfg: EnvConfig, *, seed: Optional[int] = None) -> VectorEnv:
    """Instantiate the configured environment."""
    if cfg.type == "snake":
        return ENVS.create("snake", cfg=cfg, seed=seed)
    factory = ENVS.get(cfg.type)
    return factory(
        num_envs=cfg.num_envs,
        seed=cfg.seed if seed is None else seed,
        **cfg.kwargs,
    )
