"""Schedules, injected as callables.

Keeping epsilon and beta as ``Callable[[int], float]`` means the DQN class has no
opinion about how exploration is annealed: tests pass constants, the factory
passes :class:`LinearSchedule`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["BetaFn", "Constant", "EpsilonFn", "LinearSchedule"]

EpsilonFn = Callable[[int], float]
BetaFn = Callable[[int], float]


@dataclass(frozen=True)
class LinearSchedule:
    start: float
    end: float
    steps: int

    def __call__(self, step: int) -> float:
        if self.steps <= 0:
            return self.end
        progress = min(1.0, max(0.0, step / self.steps))
        return self.start + (self.end - self.start) * progress


@dataclass(frozen=True)
class Constant:
    value: float

    def __call__(self, step: int) -> float:
        return self.value
