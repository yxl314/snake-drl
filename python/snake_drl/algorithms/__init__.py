"""Algorithms and their shared math.

Algorithms are constructed by the composition root, never by each other.  The
pure functions here (``compute_gae``, ``q_target``) are exported because they are
useful for analysis and are covered directly by tests.
"""

from __future__ import annotations

from .base import Algorithm
from .dqn import DQN, q_target
from .exploration import BetaFn, Constant, EpsilonFn, LinearSchedule
from .gae import compute_gae
from .ppo import PPO

__all__ = [
    "Algorithm",
    "BetaFn",
    "Constant",
    "DQN",
    "EpsilonFn",
    "LinearSchedule",
    "PPO",
    "compute_gae",
    "q_target",
]
