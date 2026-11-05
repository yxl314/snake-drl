"""The training loop.

It is intentionally tiny: ``while env_steps < total_steps: metrics =
algorithm.iterate()`` and every policy around that (metrics, evaluation,
checkpoints) lives in a callback.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..algorithms.base import Algorithm
from .callbacks import Callback


class Trainer:
    def __init__(
        self,
        algorithm: Algorithm,
        total_steps: int,
        callbacks: Sequence[Callback] = (),
    ) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        self.algorithm = algorithm
        self.total_steps = int(total_steps)
        self.callbacks: List[Callback] = list(callbacks)
        self.iteration = 0

    @property
    def env_steps(self) -> int:
        return self.algorithm.env_steps

    def run(self) -> "Trainer":
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

    def __enter__(self) -> "Trainer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.algorithm.close()
