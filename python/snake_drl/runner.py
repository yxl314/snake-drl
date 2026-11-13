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
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from ._dtypes import torch_dtype
from .envs import StepResult, VectorEnv

# The engine hands out read-only numpy views; torch only ever reads them.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

__all__ = ["StepBatch", "VectorRunner"]

# How many finished episodes the training curve averages over. 100 is the
# usual choice: long enough to smooth the noise of a 256-env batch, short
# enough to react within a fraction of a training run.
EPISODE_WINDOW = 100


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

        # Episode bookkeeping for the training curve. Kept here because the
        # runner is the only component that sees every reward and every done;
        # algorithms stay free of statistics code. The ring holds the most
        # recent finished episodes, which is what a smoothed curve needs.
        self._ep_return = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_length = np.zeros(self.num_envs, dtype=np.int64)
        self._finished_returns: deque[float] = deque(maxlen=EPISODE_WINDOW)
        self._finished_lengths: deque[int] = deque(maxlen=EPISODE_WINDOW)

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

        self._closed = False
        self._staging: torch.Tensor | None = None
        self._pin_reward = self._pin_terminated = self._pin_truncated = None
        if pin_memory and self.device.type == "cuda":
            self._staging = torch.empty_like(self._frame, pin_memory=True)
            self._pin_reward = torch.empty(self.num_envs, dtype=torch.float32, pin_memory=True)
            self._pin_terminated = torch.empty(self.num_envs, dtype=torch.bool, pin_memory=True)
            self._pin_truncated = torch.empty(self.num_envs, dtype=torch.bool, pin_memory=True)

    # ------------------------------------------------------------------ util
    def _copy_into(self, array: np.ndarray, dst: torch.Tensor) -> None:
        """Copy an environment array into a pre-allocated tensor.

        On CPU this goes through numpy: a torch ``copy_`` issued right after the
        engine's parallel section measured 2.07 ms for 76 KiB versus 0.094 ms
        for ``np.copyto`` (torch's intra-op threads spin up again for a copy
        that is pure memcpy).  The CUDA path keeps the pinned staging buffer.
        """
        if dst.device.type == "cpu":
            # ``view[...] = array`` applies numpy's permissive assignment cast
            # (the engine reports flags as uint8, the buffers are bool).
            dst.numpy()[...] = array
        else:
            dst.copy_(torch.from_numpy(array))

    def _upload(self, array: np.ndarray, dst: torch.Tensor) -> None:
        if self._staging is not None:
            # The single pinned staging buffer is reused within the step (next
            # frame, then final frame), so the device copy must be synchronous:
            # a non_blocking copy could still be reading it when final_obs is
            # written. The transfer is ~10 us for a 76 KiB frame.
            self._staging.copy_(torch.from_numpy(array))
            dst.copy_(self._staging)
        else:
            self._copy_into(array, dst)

    def _upload_scalar(
        self, array: np.ndarray, dst: torch.Tensor, pin: torch.Tensor | None
    ) -> None:
        if pin is not None:
            pin.copy_(torch.from_numpy(array))
            dst.copy_(pin)
        else:
            self._copy_into(array, dst)

    def _successor_frame(self, result: StepResult, done: np.ndarray) -> torch.Tensor:
        """Frame to use as the successor: final on done, next frame otherwise."""
        if result.final_obs is None:
            return self._frame_alt
        self._upload(result.final_obs, self._final)
        # Same reasoning as _copy_into: a handful of torch ops on small tensors
        # right after the engine's parallel section costs milliseconds, while
        # the numpy equivalents are microseconds.
        if self.device.type == "cpu":
            self._done.numpy()[...] = done
            self._successor.numpy()[...] = self._frame_alt.numpy()
            self._successor.numpy()[done] = self._final.numpy()[done]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._done.copy_(torch.from_numpy(done))
            self._successor.copy_(self._frame_alt)
            self._successor[self._done] = self._final[self._done]
        return self._successor

    # ------------------------------------------------------------------- api
    def reset(self) -> torch.Tensor:
        """Reset every environment and return the first window."""
        self._require_open()
        # A reset starts a new stream: partial episodes from the old one are
        # not comparable and must not leak into the curve.
        self._ep_return.fill(0.0)
        self._ep_length.fill(0)
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
        self._require_open()
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().to("cpu").numpy().astype(np.int8, copy=False)
        else:
            actions_np = np.asarray(actions, dtype=np.int8)
        if actions_np.shape != (self.num_envs,):
            raise ValueError(f"expected {self.num_envs} actions, got shape {actions_np.shape}")
        result = self.env.step(actions_np)
        done = np.logical_or(result.terminated, result.truncated)
        self._record_episodes(result.reward, done)

        obs = self.obs
        self._upload(result.obs, self._frame_alt)
        successor = self._successor_frame(result, done)

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
            if result.final_obs is not None:
                # A reset episode starts with a clean window (every slot is the
                # reset frame), matching reset(); carrying the previous
                # episode's frames into the new one would feed the encoder
                # stale history for the first `frames-1` steps.
                self._reset_window_rows(done)
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

    def _record_episodes(self, reward: np.ndarray, done: np.ndarray) -> None:
        """Accumulate per-environment episode statistics.

        Cheap by construction: two vector adds per step, and the Python loop
        only runs for the environments that finished this step. It has to
        happen on every step -- a finished episode's total is only known at
        the moment it finishes.
        """
        self._ep_return += reward
        self._ep_length += 1
        if not done.any():
            return
        for env in np.flatnonzero(done):
            self._finished_returns.append(float(self._ep_return[env]))
            self._finished_lengths.append(int(self._ep_length[env]))
        self._ep_return[done] = 0.0
        self._ep_length[done] = 0

    def episode_stats(self) -> dict[str, float]:
        """Mean return/length over the recent finished episodes.

        Returns an empty dict until at least one episode has finished, so the
        first logged record has no half-formed zeros in it.
        """
        if not self._finished_returns:
            return {}
        return {
            "train/episode_return": float(np.mean(self._finished_returns)),
            "train/episode_length": float(np.mean(self._finished_lengths)),
            "train/episodes": float(len(self._finished_returns)),
        }

    def _reset_window_rows(self, done: np.ndarray) -> None:
        """Overwrite the whole window of the finished environments."""
        if self.device.type == "cpu":
            window = self._window_alt.numpy()
            reset = self._frame_alt.numpy()
            for t in range(self.frames):
                window[done, t] = reset[done]
        else:
            mask = torch.from_numpy(done).to(self.device)
            for t in range(self.frames):
                self._window_alt[mask, t] = self._frame_alt[mask]

    @property
    def obs(self) -> torch.Tensor:
        """Current (post-step) window, ready for the next action."""
        return self._frame.unsqueeze(1) if self.frames == 1 else self._window

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.env.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("this VectorRunner has been closed")
