"""Adapter from the C++ ``VecSnake`` engine to the training :class:`VectorEnv`.

Only the dense ``grid`` observation is exposed for training.  The engine's
bitboard mode is cheaper to produce, but turning a uint64 row into a CNN input
costs an ``np.unpackbits`` round trip that measured ~11x more than simply
copying the grid observation (285 us vs 26 us for 1024 envs), so the training
path does not use it.
"""

from __future__ import annotations

import numpy as np

from .._snake_core import AutoresetMode, ObsMode, VecConfig, VecSnake
from ..config import EnvConfig
from ..registry import register
from ..spaces import ActionSpec, ObsSpec
from .base import ENVS, StepResult, VectorEnv

_RELATIVE_ACTIONS = 3  # straight, right, left
_ABSOLUTE_ACTIONS = 4  # up, right, down, left


class SnakeVecEnv(VectorEnv):
    """Batched snake games driven with the engine's ``SameStep`` autoreset."""

    def __init__(self, cfg: EnvConfig, seed: int | None = None) -> None:
        vec = VecConfig()
        vec.num_envs = cfg.num_envs
        vec.width = cfg.width
        vec.height = cfg.height
        vec.initial_length = cfg.initial_length
        vec.max_episode_steps = cfg.max_episode_steps
        vec.max_steps_without_food = cfg.max_steps_without_food
        vec.food_reward = cfg.food_reward
        vec.death_penalty = cfg.death_penalty
        vec.step_penalty = cfg.step_penalty
        vec.win_bonus = cfg.win_bonus
        vec.relative_actions = cfg.action_mode == "relative"
        vec.autoreset_mode = AutoresetMode.SameStep
        vec.obs_mode = ObsMode.Grid
        vec.seed = cfg.seed if seed is None else seed
        vec.num_threads = cfg.num_threads
        vec.parallel_threshold = cfg.parallel_threshold

        self._cfg = cfg
        self._env = VecSnake(vec)
        self._actions = np.zeros(cfg.num_envs, dtype=np.int8)
        self.num_envs = cfg.num_envs
        self.obs_spec = ObsSpec((3, cfg.height, cfg.width), "image", "uint8", 1.0)
        self.action_spec = ActionSpec(
            _RELATIVE_ACTIONS if cfg.action_mode == "relative" else _ABSOLUTE_ACTIONS
        )

    # ------------------------------------------------------------------ core
    def reset(self) -> np.ndarray:
        self._env.reset_all()
        return self._env.observations()

    def step(self, actions: np.ndarray) -> StepResult:
        self._actions[:] = actions
        self._env.step(self._actions)

        terminated = self._env.terminated()
        truncated = self._env.truncated()
        final_obs = None
        if self._env.dones().any():
            final_obs = self._env.final_observations()
        return StepResult(
            obs=self._env.observations(),
            reward=self._env.rewards(),
            terminated=terminated,
            truncated=truncated,
            final_obs=final_obs,
        )

    # --------------------------------------------------------------- extras
    def seed(self, seed: int) -> None:
        self._env.reseed_all(seed)

    def close(self) -> None:
        self._env = None  # type: ignore[assignment]

    @property
    def scores(self) -> np.ndarray:
        return self._env.scores()

    @property
    def final_scores(self) -> np.ndarray:
        return self._env.final_scores()


@register(ENVS, "snake")
def make_snake(cfg: EnvConfig, seed: int | None = None) -> SnakeVecEnv:
    return SnakeVecEnv(cfg, seed=seed)
