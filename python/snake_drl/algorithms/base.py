"""Algorithm interface.

``iterate()`` is the whole contract the trainer needs: it advances the
environment, learns, and returns the metrics of that iteration.  PPO's iterate
is "collect n_steps then run epochs"; DQN's is "collect one vectorized step then
run the credited gradient steps".  Everything else -- logging, evaluation,
checkpointing -- is a callback and stays algorithm-agnostic.

Subclasses implement ``_iterate()``; ``iterate()`` is final-ish and only keeps
the iteration counter, so the trainer can rely on it.

Counters have one meaning each:

* ``env_steps`` -- environment steps (transitions) collected;
* ``updates`` -- optimizer steps taken (PPO: minibatches, DQN: gradient steps);
* ``iterations`` -- calls to :meth:`iterate`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..nn.policies import ActorCriticPolicy, QPolicy
from ..runner import VectorRunner

__all__ = ["Algorithm"]


class Algorithm(ABC):
    policy: ActorCriticPolicy | QPolicy

    def __init__(
        self,
        policy: ActorCriticPolicy | QPolicy,
        runner: VectorRunner,
        device: torch.device,
        *,
        total_steps: int | None = None,
    ) -> None:
        self.policy = policy
        self.runner = runner
        self.device = torch.device(device)
        self.env_steps = 0
        self.updates = 0
        self.iterations = 0
        # The run horizon drives learning-rate schedules.  The factory fills it
        # from the config; the trainer asserts that both agree.
        self.total_steps = total_steps

    # ------------------------------------------------------------------ core
    def iterate(self) -> dict[str, float]:
        """One collect + learn cycle.

        Episode statistics are merged here rather than in each algorithm: the
        runner already sees every reward, and both PPO and DQN need the same
        curve.
        """
        metrics = self._iterate()
        metrics.update(self.runner.episode_stats())
        self.iterations += 1
        return metrics

    @abstractmethod
    def _iterate(self) -> dict[str, float]:
        """Implementation of :meth:`iterate`; subclasses override this."""

    # ------------------------------------------------------------ lifecycle
    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "env_steps": self.env_steps,
            "updates": self.updates,
            "iterations": self.iterations,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.env_steps = int(state.get("env_steps", 0))
        self.updates = int(state.get("updates", 0))
        self.iterations = int(state.get("iterations", 0))

    def close(self) -> None:
        self.runner.close()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(policy={type(self.policy).__name__}, "
            f"env_steps={self.env_steps}, updates={self.updates})"
        )
