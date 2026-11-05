"""Trainer callbacks.

The trainer itself only knows how to call ``iterate``; logging, evaluation and
checkpointing are callbacks, so any of them can be dropped or replaced without
touching the training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..config import ProjectConfig
from .checkpoint import prune_checkpoints, save_checkpoint
from .evaluator import Evaluator
from .logger import MetricLogger


class Callback:
    def on_start(self, trainer: "Trainer") -> None:  # noqa: F821
        pass

    def on_iteration(self, trainer: "Trainer", metrics: Dict[str, float]) -> None:  # noqa: F821
        pass

    def on_end(self, trainer: "Trainer") -> None:  # noqa: F821
        pass


class ProgressCallback(Callback):
    """tqdm over environment steps (optional; imported lazily)."""

    def __init__(self) -> None:
        self._bar = None

    def on_start(self, trainer: "Trainer") -> None:  # noqa: F821
        try:
            from tqdm import tqdm

            self._bar = tqdm(total=trainer.total_steps, initial=trainer.env_steps, unit="step")
        except ImportError:  # pragma: no cover
            self._bar = None

    def on_iteration(self, trainer: "Trainer", metrics: Dict[str, float]) -> None:  # noqa: F821
        if self._bar is not None:
            self._bar.n = trainer.env_steps
            self._bar.refresh()

    def on_end(self, trainer: "Trainer") -> None:  # noqa: F821
        if self._bar is not None:
            self._bar.close()


class LoggingCallback(Callback):
    def __init__(self, logger: MetricLogger, every: int) -> None:
        self.logger = logger
        self.every = int(every)
        self._next = self.every
        self._buffer: Dict[str, List[float]] = {}

    def on_iteration(self, trainer: "Trainer", metrics: Dict[str, float]) -> None:  # noqa: F821
        for key, value in metrics.items():
            self._buffer.setdefault(key, []).append(float(value))
        if trainer.env_steps < self._next:
            return
        averaged = {k: sum(v) / len(v) for k, v in self._buffer.items() if v}
        self.logger.log(trainer.env_steps, averaged)
        self._buffer.clear()
        self._next = trainer.env_steps + self.every

    def on_end(self, trainer: "Trainer") -> None:  # noqa: F821
        if self._buffer:
            averaged = {k: sum(v) / len(v) for k, v in self._buffer.items() if v}
            self.logger.log(trainer.env_steps, averaged)
            self._buffer.clear()


class CloseLoggerCallback(Callback):
    """Closes the logger *after* every other callback has written its final row."""

    def __init__(self, logger: MetricLogger) -> None:
        self.logger = logger

    def on_end(self, trainer: "Trainer") -> None:  # noqa: F821
        self.logger.close()


class EvalCallback(Callback):
    def __init__(self, evaluator: Evaluator, every: int, logger: Optional[MetricLogger] = None) -> None:
        self.evaluator = evaluator
        self.every = int(every)
        self.logger = logger
        self._next = self.every

    def on_iteration(self, trainer: "Trainer", metrics: Dict[str, float]) -> None:  # noqa: F821
        if trainer.env_steps < self._next:
            return
        self._next = trainer.env_steps + self.every
        results = self.evaluator.run(trainer.algorithm.policy)
        if self.logger is not None:
            self.logger.log(trainer.env_steps, results)
        else:
            print(f"[eval @ {trainer.env_steps}] " + "  ".join(f"{k}={v:.4g}" for k, v in results.items()))

    def on_end(self, trainer: "Trainer") -> None:  # noqa: F821
        results = self.evaluator.run(trainer.algorithm.policy)
        if self.logger is not None:
            self.logger.log(trainer.env_steps, results)
        else:
            print(f"[eval @ {trainer.env_steps}] " + "  ".join(f"{k}={v:.4g}" for k, v in results.items()))


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
        self.last_path: Optional[Path] = None

    def _save(self, trainer: "Trainer", name: str) -> None:  # noqa: F821
        self.last_path = save_checkpoint(
            self.run_dir / f"checkpoint_{trainer.env_steps:012d}.pt",
            trainer.algorithm,
            self.config,
        )
        prune_checkpoints(self.run_dir, self.keep_last)
        latest = self.run_dir / "latest.pt"
        latest.write_bytes(self.last_path.read_bytes())

    def on_iteration(self, trainer: "Trainer", metrics: Dict[str, float]) -> None:  # noqa: F821
        if self.every and trainer.env_steps >= self._next:
            self._next = trainer.env_steps + self.every
            self._save(trainer, "periodic")

    def on_end(self, trainer: "Trainer") -> None:  # noqa: F821
        self._save(trainer, "final")
