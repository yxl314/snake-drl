"""Trainer callbacks.

The trainer itself only knows how to call ``iterate``; logging, evaluation and
checkpointing are callbacks, so any of them can be dropped or replaced without
touching the training loop.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import ProjectConfig
from .checkpoint import prune_checkpoints, save_checkpoint
from .evaluator import Evaluator
from .logger import MetricLogger

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from .trainer import Trainer

__all__ = [
    "Callback",
    "CheckpointCallback",
    "CloseLoggerCallback",
    "EvalCallback",
    "LoggingCallback",
    "ProgressCallback",
]


class Callback:
    """Base class; override only the hooks you need."""

    def on_start(self, trainer: Trainer) -> None:
        pass

    def on_iteration(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        pass

    def on_end(self, trainer: Trainer) -> None:
        pass


class ProgressCallback(Callback):
    """tqdm over environment steps (imported lazily, so it stays optional)."""

    def __init__(self) -> None:
        self._bar = None

    def on_start(self, trainer: Trainer) -> None:
        try:
            from tqdm import tqdm

            self._bar = tqdm(total=trainer.total_steps, initial=trainer.env_steps, unit="step")
        except ImportError:  # pragma: no cover - optional dependency
            self._bar = None

    def on_iteration(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        if self._bar is not None:
            self._bar.n = trainer.env_steps
            self._bar.refresh()

    def on_end(self, trainer: Trainer) -> None:
        if self._bar is not None:
            self._bar.close()


class LoggingCallback(Callback):
    """Averages the per-iteration metrics and writes one row per interval."""

    def __init__(self, logger: MetricLogger, every: int) -> None:
        self.logger = logger
        self.every = int(every)
        self._next = self.every
        self._buffer: dict[str, list[float]] = {}

    def on_iteration(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self._buffer.setdefault(key, []).append(float(value))
        if trainer.env_steps >= self._next:
            self._flush(trainer)
            self._next = trainer.env_steps + self.every

    def on_end(self, trainer: Trainer) -> None:
        self._flush(trainer)

    def _flush(self, trainer: Trainer) -> None:
        if not self._buffer:
            return
        averaged = {key: sum(values) / len(values) for key, values in self._buffer.items()}
        self.logger.log(trainer.env_steps, averaged)
        self._buffer.clear()


class EvalCallback(Callback):
    def __init__(
        self, evaluator: Evaluator, every: int, logger: MetricLogger | None = None
    ) -> None:
        self.evaluator = evaluator
        self.every = int(every)
        self.logger = logger
        self._next = self.every

    def on_iteration(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        if trainer.env_steps >= self._next:
            self._next = trainer.env_steps + self.every
            self._evaluate(trainer)

    def on_end(self, trainer: Trainer) -> None:
        self._evaluate(trainer)

    def _evaluate(self, trainer: Trainer) -> None:
        results = self.evaluator.run(trainer.algorithm.policy)
        if self.logger is not None:
            self.logger.log(trainer.env_steps, results)
        else:
            body = "  ".join(f"{key}={value:.4g}" for key, value in results.items())
            print(f"[eval @ {trainer.env_steps}] {body}")


class CheckpointCallback(Callback):
    def __init__(
        self,
        run_dir: str | Path,
        every: int,
        config: ProjectConfig,
        keep_last: int = 3,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.every = int(every)
        self.config = config
        self.keep_last = int(keep_last)
        self._next = self.every
        self.last_path: Path | None = None

    def on_iteration(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        if self.every and trainer.env_steps >= self._next:
            self._next = trainer.env_steps + self.every
            self._save(trainer)

    def on_end(self, trainer: Trainer) -> None:
        self._save(trainer)

    def _save(self, trainer: Trainer) -> None:
        path = save_checkpoint(
            self.run_dir / f"checkpoint_{trainer.env_steps:012d}.pt",
            trainer.algorithm,
            self.config,
        )
        # Stream instead of reading the whole file into memory twice.
        shutil.copyfile(path, self.run_dir / "latest.pt")
        self.last_path = path
        prune_checkpoints(self.run_dir, self.keep_last)


class CloseLoggerCallback(Callback):
    """Closes the logger *after* every other callback wrote its final row."""

    def __init__(self, logger: MetricLogger) -> None:
        self.logger = logger

    def on_end(self, trainer: Trainer) -> None:
        self.logger.close()
