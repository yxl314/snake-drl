"""The only component that touches the environment.

Responsibilities:
  * maintain the per-environment observation window the encoder expects;
  * move frames to the compute device through a pinned staging buffer when CUDA
    is available, so the copy can overlap;
  * report the *true* successor observation for terminal/truncated steps, which
    is what makes PPO's GAE and the DQN target bootstrap correctly;
  * hand out views instead of copies -- the consumer writes them into its own
    buffer, which is where the copy has to happen anyway.

Contract: returned tensors alias internal storage and are only valid until the
next :meth:`step`/:meth:`reset`.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from .envs import VectorEnv
from .spaces import StepBatch

# The engine hands out read-only numpy views; torch only ever reads them.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

_DTYPES = {"uint8": torch.uint8, "float32": torch.float32}


class VectorRunner:
    def __init__(
        self,
        env: VectorEnv,
        frames: int,
        device: torch.device | str,
        *,
        pin_memory: bool = True,
    ) -> None:
        self.env = env
        self.frames = int(frames)
        self.device = torch.device(device)
        self.num_envs = env.num_envs
        self.obs_spec = env.obs_spec
        self.action_spec = env.action_spec

        dtype = _DTYPES[env.obs_spec.dtype]
        window_shape = (self.num_envs, self.frames, *env.obs_spec.shape)
        frame_shape = (self.num_envs, *env.obs_spec.shape)
        self._window = torch.empty(window_shape, dtype=dtype, device=self.device)
        self._window_alt = torch.empty_like(self._window)
        self._next = torch.empty_like(self._window)
        self._frame = torch.empty(frame_shape, dtype=dtype, device=self.device)
        self._final = torch.empty_like(self._frame)
        self._succ = torch.empty_like(self._frame)

        # Small per-step scalars are staged as well; letting them trigger a
        # synchronous copy would serialise the GPU pipeline every step.
        self._reward = torch.empty(self.num_envs, dtype=torch.float32, device=self.device)
        self._terminated = torch.empty(self.num_envs, dtype=torch.bool, device=self.device)
        self._truncated = torch.empty(self.num_envs, dtype=torch.bool, device=self.device)
        self._actions = np.zeros(self.num_envs, dtype=np.int8)
        self._broadcast = (self.num_envs, *([1] * len(env.obs_spec.shape)))

        self._staging = None
        self._staging_small = None
        self._pin_reward = self._pin_terminated = self._pin_truncated = None
        if pin_memory and self.device.type == "cuda":
            self._staging = torch.empty_like(self._frame, pin_memory=True)
            self._staging_small = torch.empty(2 * self.num_envs, dtype=torch.uint8, pin_memory=True)
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

    def _upload_scalar(self, array: np.ndarray, dst: torch.Tensor, pin: torch.Tensor | None) -> None:
        source = torch.from_numpy(array)
        if pin is not None:
            pin.copy_(source)
            dst.copy_(pin, non_blocking=True)
        else:
            dst.copy_(source)

    # ------------------------------------------------------------------- api
    def reset(self) -> torch.Tensor:
        obs = self.env.reset()
        frame = torch.from_numpy(obs)
        self._window[:, 0].copy_(frame)
        for t in range(1, self.frames):
            self._window[:, t].copy_(self._window[:, 0])
        return self._window

    def step(self, actions: torch.Tensor | np.ndarray) -> StepBatch:
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().to("cpu").numpy().astype(np.int8, copy=False)
        else:
            actions_np = np.asarray(actions, dtype=np.int8)
        result = self.env.step(actions_np)

        obs = self._window  # pre-step window; replaced, never mutated in place
        self._upload(result.obs, self._frame)
        if result.final_obs is not None:
            self._upload(result.final_obs, self._final)
            done = np.logical_or(result.terminated, result.truncated)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                done_t = torch.from_numpy(done).to(self.device)
            self._succ.copy_(self._frame)
            self._succ[done_t] = self._final[done_t]
            successor = self._succ
        else:
            successor = self._frame

        if self.frames == 1:
            next_obs = successor.unsqueeze(1)
        else:
            self._next[:, :-1].copy_(obs[:, 1:])
            self._next[:, -1].copy_(successor)
            next_obs = self._next

        # Advance the environment window with the *reset* frame, not the final
        # one: the following transition must observe the new episode.  Writing
        # into an alternate buffer keeps ``obs`` aliasing the old window.
        if self.frames > 1:
            self._window_alt[:, :-1].copy_(obs[:, 1:])
        self._window_alt[:, -1].copy_(self._frame)
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
        """Current (post-step) observation window, ready for the next action."""
        return self._window

    def close(self) -> None:
        self.env.close()
