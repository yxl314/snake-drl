"""Vectorized environment interface.

Deliberately numpy-only: the environment layer knows nothing about torch, the
algorithms or the buffers.  An environment is a batch of ``N`` independent games
stepped in lockstep, with the same ``terminated/truncated`` split gymnasium uses.

Autoreset convention (identical to the engine): the step that ends an episode
also resets it, so every step produces a usable transition.  The final
observation of the finished episode is reported separately as
``StepResult.final_obs`` -- that is what makes bootstrapping on truncation
correct.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..registry import Registry
from ..spaces import ActionSpec, ObsSpec


@dataclass(frozen=True)
class StepResult:
    """One vectorized step.

    Every array is either a zero-copy view into environment memory (invalidated
    by the next call) or a freshly allocated array.  Consumers must copy what
    they need to keep.
    """

    obs: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    final_obs: Optional[np.ndarray]


class VectorEnv(ABC):
    """The contract every environment implements."""

    num_envs: int
    obs_spec: ObsSpec
    action_spec: ActionSpec

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset every environment and return the first observation batch."""

    @abstractmethod
    def step(self, actions: np.ndarray) -> StepResult:
        """Apply one action per environment."""

    def close(self) -> None:
        """Release engine resources (no-op by default)."""

    def __enter__(self) -> "VectorEnv":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


ENVS: Registry[Callable[..., VectorEnv]] = Registry("environment")
