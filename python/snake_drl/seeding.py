"""Deterministic seeding helpers.

Hash randomisation is *not* touched here: ``PYTHONHASHSEED`` only takes effect
before the interpreter starts, so setting it at import time would be misleading.
"""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["derive_seed", "seed_all"]


def seed_all(seed: int, *, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (including CUDA) from one integer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cudnn.benchmark is disabled by configure_runtime on CUDA; these are the
        # remaining reproducible-XP knobs.
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def derive_seed(base: int, stream: int) -> int:
    """Independent, reproducible per-stream seed (train env / eval env / buffer)."""
    return (int(base) * 0x9E3779B1 + int(stream) * 0x85EBCA77 + 0x27D4EB2F) & 0x7FFFFFFF
