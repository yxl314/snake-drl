"""Algorithm interface.

``iterate()`` is the whole contract the trainer needs: it advances the
environment, learns, and returns the metrics of that iteration.  PPO's iterate
is "collect n_steps then run epochs"; DQN's is "collect one vectorized step then
run gradient_steps updates".  Everything else (logging, evaluation, checkpointing)
is a callback and stays algorithm-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn

from ..runner import VectorRunner


class Algorithm(ABC):
    def __init__(self, policy: nn.Module, runner: VectorRunner, device: torch.device) -> None:
        self.policy = policy
        self.runner = runner
        self.device = device
        self.env_steps = 0
        self.updates = 0
        self.total_steps: int | None = None  # set by the factory for schedules

    # ------------------------------------------------------------------ core
    @abstractmethod
    def iterate(self) -> Dict[str, float]:
        """One collect + learn cycle."""

    # ------------------------------------------------------------ lifecycle
    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "env_steps": self.env_steps,
            "updates": self.updates,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.env_steps = int(state.get("env_steps", 0))
        self.updates = int(state.get("updates", 0))

    @property
    def n_actions(self) -> int:
        return self.runner.action_spec.n

    @property
    def frames(self) -> int:
        return self.runner.frames

    def close(self) -> None:
        self.runner.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(policy={type(self.policy).__name__}, env_steps={self.env_steps})"
