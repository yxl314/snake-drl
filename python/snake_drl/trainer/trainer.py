"""The training loop.

It is intentionally tiny: ``while env_steps < total_steps: metrics =
algorithm.iterate()`` and everything around that (metrics, evaluation,
checkpoints) lives in a callback.

The trainer owns the run horizon: it is the single source of truth for
``algorithm.total_steps``, which drives learning-rate schedules.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..algorithms.base import Algorithm
from .callbacks import Callback

__all__ = ["Trainer"]


class Trainer:
    def __init__(
        self,
        algorithm: Algorithm,
        total_steps: int,
        callbacks: Sequence[Callback] = (),
    ) -> None:
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        if algorithm.total_steps is None:
            algorithm.total_steps = total_steps
        elif algorithm.total_steps != total_steps:
            raise ValueError(
                f"algorithm was built for {algorithm.total_steps} steps but the trainer "
                f"runs {total_steps}; build both from the same config"
            )
        self.algorithm = algorithm
        self.total_steps = int(total_steps)
        self.callbacks: list[Callback] = list(callbacks)
        self.iteration = 0

    @property
    def env_steps(self) -> int:
        return self.algorithm.env_steps

    def run(self) -> Trainer:
        for callback in self.callbacks:
            callback.on_start(self)
        while self.algorithm.env_steps < self.total_steps:
            metrics = self.algorithm.iterate()
            self.iteration += 1
            for callback in self.callbacks:
                callback.on_iteration(self, metrics)
        for callback in self.callbacks:
            callback.on_end(self)
        return self

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.algorithm.close()
