"""Deterministic seeding helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_all(seed: int, *, deterministic: bool = False) -> None:
    """Seed python/numpy/torch (and CUDA) from one integer."""
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def derive_seed(base: int, stream: int) -> int:
    """Independent, reproducible per-stream seed (train env / eval env / init)."""
    return (int(base) * 0x9E3779B1 + int(stream) * 0x85EBCA77 + 0x27D4EB2F) & 0x7FFFFFFF


def make_rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)
