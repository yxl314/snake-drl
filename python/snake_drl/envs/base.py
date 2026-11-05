"""Vectorized environment interface.

This layer is numpy-only: it knows nothing about torch, algorithms or buffers.
An environment is a batch of ``N`` independent games stepped in lockstep, with
the same ``terminated``/``truncated`` split gymnasium uses.

Autoreset convention (identical to the engine): the step that ends an episode
also resets it, so every step produces a usable transition.  The final
observation of the finished episode is reported separately as
``StepResult.final_obs`` -- that is what makes bootstrapping on truncation
correct.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..spaces import ActionSpec, ObsSpec

__all__ = ["StepResult", "VectorEnv"]


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
    final_obs: np.ndarray | None


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

    def close(self) -> None:  # noqa: B027 - optional hook, safe to leave empty
        """Release resources. Must be safe to call more than once."""

    def __enter__(self) -> VectorEnv:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
