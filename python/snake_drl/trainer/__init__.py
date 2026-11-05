"""Training orchestration: runner service, evaluation, logging, composition root."""

from __future__ import annotations

from .callbacks import (
    Callback,
    CheckpointCallback,
    CloseLoggerCallback,
    EvalCallback,
    LoggingCallback,
    ProgressCallback,
)
from .checkpoint import load_checkpoint, prune_checkpoints, save_checkpoint
from .evaluator import Evaluator
from .factory import build_algorithm, build_evaluator, build_trainer, load_policy
from .logger import MetricLogger
from .trainer import Trainer

__all__ = [
    "Callback",
    "CheckpointCallback",
    "CloseLoggerCallback",
    "EvalCallback",
    "Evaluator",
    "LoggingCallback",
    "MetricLogger",
    "ProgressCallback",
    "Trainer",
    "build_algorithm",
    "build_evaluator",
    "build_trainer",
    "load_checkpoint",
    "load_policy",
    "prune_checkpoints",
    "save_checkpoint",
]
