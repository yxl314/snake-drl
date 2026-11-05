"""Training orchestration: runner, evaluator, logger, callbacks, composition root."""

from .callbacks import (
    Callback,
    CheckpointCallback,
    CloseLoggerCallback,
    EvalCallback,
    LoggingCallback,
    ProgressCallback,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .evaluator import Evaluator
from .factory import (
    build_algorithm,
    build_evaluator,
    build_policy_for,
    build_trainer,
    load_policy,
    policy_family,
)
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
    "build_policy_for",
    "build_trainer",
    "load_checkpoint",
    "load_policy",
    "policy_family",
    "save_checkpoint",
]
