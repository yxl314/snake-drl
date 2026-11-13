"""A trained policy driving one environment, for demo and recording.

Human play uses :class:`~snake_drl.play.game.GameSession`: absolute-direction
actions, no time limit, terminal state kept on screen. A trained agent needs the
opposite -- exactly the environment it was trained on -- so this is a separate
session with the same small surface the Qt window needs.

One deliberate difference from training: autoreset is forced to ``Disabled`` so
the terminal state stays on screen instead of being replaced inside the same
step. That only changes what happens *after* death, which the demo never steps
past.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..device import resolve_device
from ..runner import VectorRunner
from ..trainer.factory import load_policy

__all__ = ["AgentSession"]


class AgentSession:
    """One environment driven by a checkpoint's greedy policy.

    The session reuses the checkpoint's own ``env`` section (board size,
    observation mode, action mode) so the policy sees the distribution it was
    trained on; ``num_envs`` is forced to 1 and autoreset to ``Disabled``.

    Args:
        checkpoint: Path to a ``.pt`` checkpoint written by the trainer.
        device: ``cpu``, ``cuda`` or ``auto``.
        seed: Overrides the checkpoint's environment seed when given.

    Attributes:
        width (int): Board width, taken from the checkpoint.
        height (int): Board height, taken from the checkpoint.
    """

    def __init__(self, checkpoint: str, device: str = "auto", *, seed: int | None = None) -> None:
        from ..envs.snake import make_snake_demo

        self.device = resolve_device(device)
        self.policy, self.config, _payload = load_policy(checkpoint, self.device)
        env_cfg = replace(self.config.env, num_envs=1)
        if seed is not None:
            env_cfg = replace(env_cfg, seed=seed)
        self.width = int(env_cfg.width)
        self.height = int(env_cfg.height)
        # Disabled autoreset keeps the terminal frame on screen; training uses
        # SameStep, which replaces the episode inside the killing step.
        self._env = make_snake_demo(env_cfg, seed=env_cfg.seed)
        self._runner = VectorRunner(self._env, self.policy.frames, self.device)
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Start a new episode."""
        self._obs = self._runner.reset()
        # Read the engine's real starting values instead of assuming zero: the
        # snake begins at ``initial_length``, and a HUD showing length=0 for one
        # frame looks like a bug.
        engine = self._engine()
        self.score = int(engine.scores()[0]) if engine is not None else 0
        self.length = int(engine.lengths()[0]) if engine is not None else 0
        self.steps = 0
        self.game_over = False
        self.won = False
        self.last_reward = 0.0

    @property
    def board(self) -> np.ndarray:
        """``(3, H, W)`` uint8 view: body, head and food planes of the newest frame."""
        frame = self._obs
        # obs is (B, T, C, H, W); the newest frame is the one on screen.
        while frame.dim() > 4:
            frame = frame[:, -1]
        return frame[0].detach().to("cpu").numpy().astype(np.uint8, copy=False)

    @property
    def direction_name(self) -> str:
        """The agent drives; a heading label would be noise, so it stays empty."""
        return ""

    @property
    def episode_label(self) -> str:
        """One-line status for the window's HUD.

        Chinese, matching the rest of the Qt front-end (`得分 …` in human mode).
        """
        return f"得分 {self.score}    长度 {self.length}    步数 {self.steps}"

    # ------------------------------------------------------------------ moves

    def step(self) -> bool:
        """Advance one tick with the greedy action. Returns False once finished."""
        if self.game_over:
            return False
        import torch

        with torch.no_grad():
            action = self.policy.greedy(self._obs)
        batch = self._runner.step(action)
        self.steps += 1
        self.last_reward = float(batch.reward[0])
        self._obs = self._runner.obs
        engine = self._engine()
        if engine is not None:
            self.score = int(engine.scores()[0])
            self.length = int(engine.lengths()[0])
        if bool(batch.terminated[0]) or bool(batch.truncated[0]):
            self.game_over = True
            self.won = bool(engine.wins()[0]) if engine is not None else False
        return True

    # ------------------------------------------------------------- internal

    def _engine(self):  # pragma: no cover - thin accessor kept for the HUD
        return getattr(self._env, "_engine", None)

    def close(self) -> None:
        """Release the environment and its thread pool."""
        self._env.close()
