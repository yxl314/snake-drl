"""Deterministic toy environments for tests and algorithm sanity checks.

They implement the same autoreset semantics as the engine, so the runner,
buffers and algorithms can be exercised without the C++ extension.  Both are
Markov in the one-hot position only -- no hidden counters or step-dependent
rules, which matters because a non-Markov test MDP silently breaks Q-learning.

They are registered like any other environment, so a config can select them:

.. code-block:: yaml

    env: {type: corridor, num_envs: 16, max_episode_steps: 30,
          kwargs: {length: 6}}
"""

from __future__ import annotations

import numpy as np

from ..config import EnvConfig
from ..registry import register
from ..spaces import ActionSpec, ObsSpec
from .base import StepResult, VectorEnv
from .registry import ENVS

__all__ = ["BanditEnv", "CorridorEnv", "make_bandit", "make_corridor"]


class _NumpyEnv(VectorEnv):
    """Shared bookkeeping for the numpy-only toy environments."""

    def __init__(self, num_envs: int, seed: int = 0) -> None:
        self.num_envs = int(num_envs)
        self._rng = np.random.default_rng(seed)
        self._terminated = np.zeros(self.num_envs, dtype=bool)
        self._truncated = np.zeros(self.num_envs, dtype=bool)
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"this {type(self).__name__} has been closed")


class BanditEnv(_NumpyEnv):
    """One-step contextual bandit: reward 1.0 iff the action matches the context.

    The optimal policy is ``argmax obs`` and Q* is 1.0 for the correct action and
    0.0 otherwise, which makes it a fast correctness test for PPO and the DQN
    family alike.
    """

    def __init__(self, num_envs: int = 8, n_ctx: int = 3, seed: int = 0) -> None:
        self.n_ctx = int(n_ctx)
        self.obs_spec = ObsSpec((self.n_ctx,), kind="vector", dtype="float32")
        self.action_spec = ActionSpec(self.n_ctx)
        super().__init__(num_envs, seed)
        self._ctx = np.zeros(self.num_envs, dtype=np.int64)

    def _obs(self, ctx: np.ndarray) -> np.ndarray:
        out = np.zeros((self.num_envs, self.n_ctx), dtype=np.float32)
        out[np.arange(self.num_envs), ctx] = 1.0
        return out

    def reset(self) -> np.ndarray:
        self._require_open()
        self._ctx = self._rng.integers(0, self.n_ctx, self.num_envs)
        return self._obs(self._ctx)

    def step(self, actions: np.ndarray) -> StepResult:
        self._require_open()
        ctx = self._ctx
        reward = (np.asarray(actions) == ctx).astype(np.float32)
        final = self._obs(ctx)
        self._ctx = self._rng.integers(0, self.n_ctx, self.num_envs)
        obs = self._obs(self._ctx)
        self._terminated[:] = True  # one-step episodes
        self._truncated[:] = False
        return StepResult(obs, reward, self._terminated, self._truncated, final)


class CorridorEnv(_NumpyEnv):
    """1-D corridor: action 1 moves right, action 0 moves left.

    Reaching the right end terminates with +1.0; walking off the left end
    terminates with -1.0.  Every transition depends only on ``(position,
    action)``, so the MDP is fully observable from the one-hot position, and it
    is acyclic: values stay in [-1, 1] and bootstrapping cannot inflate.  The
    optimal policy is "always right" (return 1.0).  ``max_episode_steps``
    produces genuine truncations for bootstrap tests.
    """

    def __init__(
        self, num_envs: int = 8, length: int = 6, max_episode_steps: int = 0, seed: int = 0
    ) -> None:
        if length < 2:
            raise ValueError(f"corridor length must be >= 2, got {length}")
        self.length = int(length)
        self.max_episode_steps = int(max_episode_steps)
        self.obs_spec = ObsSpec((self.length,), kind="vector", dtype="float32")
        self.action_spec = ActionSpec(2)
        super().__init__(num_envs, seed)
        self._pos = np.zeros(self.num_envs, dtype=np.int64)
        self._steps = np.zeros(self.num_envs, dtype=np.int64)

    def _obs(self) -> np.ndarray:
        out = np.zeros((self.num_envs, self.length), dtype=np.float32)
        out[np.arange(self.num_envs), self._pos] = 1.0
        return out

    def reset(self) -> np.ndarray:
        self._require_open()
        self._pos[:] = 0
        self._steps[:] = 0
        return self._obs()

    def step(self, actions: np.ndarray) -> StepResult:
        self._require_open()
        actions = np.asarray(actions)
        obs_before = self._obs()
        walking_off = (actions == 0) & (self._pos == 0)
        self._pos = np.clip(self._pos + np.where(actions == 1, 1, -1), 0, self.length - 1)
        self._steps += 1

        reward = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        reward[self._pos == self.length - 1] = 1.0
        terminated[self._pos == self.length - 1] = True
        reward[walking_off] = -1.0
        terminated[walking_off] = True

        truncated = np.zeros(self.num_envs, dtype=bool)
        if self.max_episode_steps:
            truncated = (self._steps >= self.max_episode_steps) & ~terminated

        done = terminated | truncated
        final = obs_before
        if done.any():
            final = self._obs()
            final[~done] = obs_before[~done]
            self._pos[done] = 0
            self._steps[done] = 0
        return StepResult(self._obs(), reward, terminated, truncated, final)


@register(ENVS, "bandit")
def make_bandit(cfg: EnvConfig, *, seed: int | None = None) -> BanditEnv:
    return BanditEnv(
        num_envs=cfg.num_envs,
        n_ctx=int(cfg.kwargs.get("n_ctx", 3)),
        seed=cfg.seed if seed is None else seed,
    )


@register(ENVS, "corridor")
def make_corridor(cfg: EnvConfig, *, seed: int | None = None) -> CorridorEnv:
    return CorridorEnv(
        num_envs=cfg.num_envs,
        length=int(cfg.kwargs.get("length", 6)),
        max_episode_steps=cfg.max_episode_steps,
        seed=cfg.seed if seed is None else seed,
    )
