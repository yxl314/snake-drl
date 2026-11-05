"""The only component that touches the environment.

Responsibilities:
  * maintain the per-environment observation window the encoder expects;
  * move frames to the compute device through a pinned staging buffer when CUDA
    is available, so the transfer can overlap with the engine's next step;
  * report the *true* successor observation of terminal/truncated steps, which
    is what makes PPO's GAE and the DQN target bootstrap correctly;
  * hand out views instead of copies -- the consumer writes them into its own
    buffer, which is where the copy has to happen anyway.

For a single-frame window (the common CNN/MLP case) the current frame lives in
one of two ping-pong buffers and is swapped in, so a step costs one upload and
no window maintenance at all.  Copying through a ``(B, 1, C, H, W)`` slice
instead measured ~2x the engine step itself for 256 environments.

Contract: returned tensors alias internal storage and stay valid only until the
next :meth:`step`/:meth:`reset`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch

from ._dtypes import torch_dtype
from .envs import VectorEnv

# The engine hands out read-only numpy views; torch only ever reads them.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

__all__ = ["StepBatch", "VectorRunner"]


@dataclass(frozen=True)
class StepBatch:
    """One vectorized environment step, as device tensors.

    ``obs`` is the observation that *produced* ``action``; ``next_obs`` is the
    true successor observation.  For terminal and truncated steps ``next_obs``
    is the final observation reported by the environment, not the reset state --
    that is what makes truncation bootstrapping correct.
    """

    obs: torch.Tensor  # (B, T, *frame_shape)
    next_obs: torch.Tensor
    reward: torch.Tensor  # (B,) float32
    terminated: torch.Tensor  # (B,) bool
    truncated: torch.Tensor  # (B,) bool


class VectorRunner:
    """Drives a :class:`VectorEnv` and keeps the observation window in sync."""

    def __init__(
        self,
        env: VectorEnv,
        frames: int,
        device: torch.device | str,
        *,
        pin_memory: bool = True,
    ) -> None:
        if frames < 1:
            raise ValueError(f"frames must be >= 1, got {frames}")
        self.env = env
        self.frames = int(frames)
        self.device = torch.device(device)
        self.num_envs = env.num_envs
        self.obs_spec = env.obs_spec
        self.action_spec = env.action_spec

        dtype = torch_dtype(env.obs_spec)
        frame_shape = (self.num_envs, *env.obs_spec.shape)
        self._frame = torch.empty(frame_shape, dtype=dtype, device=self.device)
        self._frame_alt = torch.empty_like(self._frame)
        self._final = torch.empty_like(self._frame)
        self._successor = torch.empty_like(self._frame)

        if self.frames > 1:
            window_shape = (self.num_envs, self.frames, *env.obs_spec.shape)
            self._window = torch.empty(window_shape, dtype=dtype, device=self.device)
            self._window_alt = torch.empty_like(self._window)
            self._next_window = torch.empty_like(self._window)

        # Per-step scalars are staged too: an un-pinned copy would synchronize
        # the device and serialize the pipeline on every step.
        self._reward = torch.empty(self.num_envs, dtype=torch.float32, device=self.device)
        self._terminated = torch.empty(self.num_envs, dtype=torch.bool, device=self.device)
        self._truncated = torch.empty(self.num_envs, dtype=torch.bool, device=self.device)
        self._done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._staging: torch.Tensor | None = None
        self._pin_reward = self._pin_terminated = self._pin_truncated = None
        if pin_memory and self.device.type == "cuda":
            self._staging = torch.empty_like(self._frame, pin_memory=True)
            self._pin_reward = torch.empty(self.num_envs, dtype=torch.float32, pin_memory=True)
            self._pin_terminated = torch.empty(self.num_envs, dtype=torch.bool, pin_memory=True)
            self._pin_truncated = torch.empty(self.num_envs, dtype=torch.bool, pin_memory=True)

    # ------------------------------------------------------------------ util
    def _upload(self, array: np.ndarray, dst: torch.Tensor) -> None:
        source = torch.from_numpy(array)
        if self._staging is not None:
            self._staging.copy_(source)
            dst.copy_(self._staging, non_blocking=True)
        else:
            dst.copy_(source)

    def _upload_scalar(
        self, array: np.ndarray, dst: torch.Tensor, pin: torch.Tensor | None
    ) -> None:
        source = torch.from_numpy(array)
        if pin is not None:
            pin.copy_(source)
            dst.copy_(pin, non_blocking=True)
        else:
            dst.copy_(source)

    def _successor_frame(self, result) -> torch.Tensor:
        """Frame to use as the successor: final on done, next frame otherwise."""
        if result.final_obs is None:
            return self._frame_alt
        self._upload(result.final_obs, self._final)
        done = np.logical_or(result.terminated, result.truncated)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._done.copy_(torch.from_numpy(done))
        self._successor.copy_(self._frame_alt)
        self._successor[self._done] = self._final[self._done]
        return self._successor

    # ------------------------------------------------------------------- api
    def reset(self) -> torch.Tensor:
        """Reset every environment and return the first window."""
        frame = torch.from_numpy(self.env.reset())
        if self.frames == 1:
            self._frame.copy_(frame)
            return self.obs
        # A fresh episode has no history: repeat the first frame across the
        # window instead of showing zeros.
        self._window[:, 0].copy_(frame)
        for t in range(1, self.frames):
            self._window[:, t].copy_(self._window[:, 0])
        return self._window

    def step(self, actions: torch.Tensor | np.ndarray) -> StepBatch:
        """Advance every environment by one step; returns views, not copies."""
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().to("cpu").numpy().astype(np.int8, copy=False)
        else:
            actions_np = np.asarray(actions, dtype=np.int8)
        if actions_np.shape != (self.num_envs,):
            raise ValueError(f"expected {self.num_envs} actions, got shape {actions_np.shape}")
        result = self.env.step(actions_np)

        obs = self.obs
        self._upload(result.obs, self._frame_alt)
        successor = self._successor_frame(result)

        if self.frames == 1:
            next_obs = successor.unsqueeze(1)
            # Ping-pong: the freshly uploaded frame becomes the current window.
            self._frame, self._frame_alt = self._frame_alt, self._frame
        else:
            self._next_window[:, :-1].copy_(obs[:, 1:])
            self._next_window[:, -1].copy_(successor)
            next_obs = self._next_window

            # Advance the environment window with the *reset* frame, not the
            # final one: the next transition must observe the new episode.
            # Writing into an alternate buffer keeps ``obs`` aliasing the old
            # window for the caller.
            self._window_alt[:, :-1].copy_(obs[:, 1:])
            self._window_alt[:, -1].copy_(self._frame_alt)
            self._window, self._window_alt = self._window_alt, self._window

        self._upload_scalar(result.reward, self._reward, self._pin_reward)
        self._upload_scalar(result.terminated, self._terminated, self._pin_terminated)
        self._upload_scalar(result.truncated, self._truncated, self._pin_truncated)
        return StepBatch(
            obs=obs,
            next_obs=next_obs,
            reward=self._reward,
            terminated=self._terminated,
            truncated=self._truncated,
        )

    @property
    def obs(self) -> torch.Tensor:
        """Current (post-step) window, ready for the next action."""
        return self._frame.unsqueeze(1) if self.frames == 1 else self._window

    def close(self) -> None:
        self.env.close()
