"""Checkpoint save/load.

A checkpoint carries everything needed to resume *and* to rebuild the policy for
evaluation: the config, the policy state, the optimizer state and the counters.
``torch.load(..., map_location=...)`` keeps CPU/GPU portable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..algorithms.base import Algorithm
from ..config import ProjectConfig, to_dict

__all__ = ["load_checkpoint", "prune_checkpoints", "save_checkpoint"]


def save_checkpoint(
    path: str | Path,
    algorithm: Algorithm,
    config: ProjectConfig,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a self-describing checkpoint and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": to_dict(config),
            "algorithm": algorithm.state_dict(),
            "env_steps": algorithm.env_steps,
            "updates": algorithm.updates,
            "iterations": algorithm.iterations,
            "extra": extra or {},
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def prune_checkpoints(directory: str | Path, keep_last: int) -> None:
    """Keep the ``keep_last`` most recent ``checkpoint_*.pt`` files."""
    if keep_last <= 0:
        return
    files = sorted(Path(directory).glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime)
    for stale in files[:-keep_last]:
        stale.unlink(missing_ok=True)
