"""Single-environment game session used by the human play mode.

Kept free of Qt imports so it can be unit tested or scripted headlessly.
"""

from __future__ import annotations

import numpy as np

from .._snake_core import AutoresetMode, ObsMode, VecConfig, VecSnake

# Direction indices match the engine's kDx/kDy tables: up, right, down, left.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

_OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

_DIRECTION_NAMES = {UP: "上", RIGHT: "右", DOWN: "下", LEFT: "左"}


class GameSession:
    """One snake game driven by absolute-direction actions."""

    def __init__(
        self,
        width: int = 16,
        height: int = 16,
        *,
        initial_length: int = 3,
        seed: int | None = None,
    ) -> None:
        cfg = VecConfig()
        cfg.num_envs = 1
        cfg.width = width
        cfg.height = height
        cfg.initial_length = initial_length
        cfg.obs_mode = ObsMode.Grid
        # Disabled keeps the terminal state on screen instead of resetting it
        # inside the step that kills the snake.
        cfg.autoreset_mode = AutoresetMode.Disabled
        cfg.relative_actions = False  # actions are absolute directions
        cfg.max_episode_steps = 0  # no time limit for human play
        cfg.max_steps_without_food = 0
        cfg.num_threads = 1
        if seed is not None:
            cfg.seed = seed
        self._cfg = cfg
        self.env = VecSnake(cfg)
        # The Grid buffer is allocated once by the constructor; holding the
        # view keeps rendering allocation-free and shares engine memory.
        self._obs_view = self.env.observations()
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        self.env.reset_all()
        self._pending_direction: int | None = None
        self.direction = int(self.env.directions()[0])
        self.score = 0
        self.length = self._cfg.initial_length
        self.steps = 0
        self.game_over = False
        self.won = False
        self.last_reward = 0.0

    @property
    def board(self) -> np.ndarray:
        """(3, H, W) uint8: plane 0 body, plane 1 head, plane 2 food."""
        return self._obs_view[0]

    @property
    def direction_name(self) -> str:
        return _DIRECTION_NAMES[self.direction]

    # ------------------------------------------------------------------ moves

    def turn(self, direction: int) -> bool:
        """Queue an absolute direction. Returns False if it was rejected."""
        if direction not in (UP, RIGHT, DOWN, LEFT):
            raise ValueError(f"invalid direction: {direction}")
        if self.game_over:
            return False
        # Validate against the engine's actual heading, not the last requested
        # direction: several turns may arrive between two ticks.
        current = int(self.env.directions()[0])
        if direction == _OPPOSITE[current]:
            return False
        self._pending_direction = direction
        changed = direction != current
        self.direction = direction
        return changed

    def step(self) -> bool:
        """Advance one tick. Returns False when the game is already over."""
        if self.game_over:
            return False

        action = self.direction if self._pending_direction is None else self._pending_direction
        self.env.step(np.array([action], dtype=np.int8))
        self._pending_direction = None

        self.direction = int(self.env.directions()[0])
        self.score = int(self.env.scores()[0])
        self.length = int(self.env.lengths()[0])
        self.steps = int(self.env.steps()[0])
        self.last_reward = float(self.env.rewards()[0])

        if self.env.dones()[0]:
            self.game_over = True
            self.won = bool(self.env.wins()[0])
        return True
