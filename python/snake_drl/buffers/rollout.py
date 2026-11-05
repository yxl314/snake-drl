"""Pre-allocated on-policy rollout storage.

The buffer owns every tensor it needs, so a collection step is a ``copy_`` into
a slice and the update step works on flattened *views* -- no per-step allocation
and no re-shuffle copies (minibatches are slices of one ``randperm``).

Memory is ``steps * num_envs * frames * frame_bytes`` for each of ``obs`` and
``next_obs``; storing ``next_obs`` is what makes truncation bootstrapping exact.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..spaces import ObsSpec

_TORCH_DTYPES = {"uint8": torch.uint8, "float32": torch.float32}


class RolloutBuffer:
    def __init__(
        self,
        steps: int,
        num_envs: int,
        obs_spec: ObsSpec,
        frames: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if steps < 1 or num_envs < 1:
            raise ValueError("steps and num_envs must be >= 1")
        self.steps = int(steps)
        self.num_envs = int(num_envs)
        self.frames = int(frames)
        self.device = torch.device(device)
        dtype = _TORCH_DTYPES[obs_spec.dtype]
        window = (self.frames, *obs_spec.shape)

        self.obs = torch.empty((steps, num_envs, *window), dtype=dtype, device=self.device)
        self.next_obs = torch.empty_like(self.obs)
        self.actions = torch.empty((steps, num_envs), dtype=torch.int64, device=self.device)
        self.rewards = torch.empty((steps, num_envs), dtype=torch.float32, device=self.device)
        self.logprobs = torch.empty((steps, num_envs), dtype=torch.float32, device=self.device)
        self.values = torch.empty((steps, num_envs), dtype=torch.float32, device=self.device)
        self.terminated = torch.empty((steps, num_envs), dtype=torch.bool, device=self.device)
        self.truncated = torch.empty((steps, num_envs), dtype=torch.bool, device=self.device)
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    @property
    def cursor(self) -> int:
        return self._cursor

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        logprob: torch.Tensor,
        value: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        t = self._cursor
        if t >= self.steps:
            raise RuntimeError(f"rollout buffer is full ({self.steps} steps); call reset()")
        self.obs[t].copy_(obs)
        self.next_obs[t].copy_(next_obs)
        self.actions[t].copy_(action)
        self.rewards[t].copy_(reward)
        self.logprobs[t].copy_(logprob)
        self.values[t].copy_(value)
        self.terminated[t].copy_(terminated)
        self.truncated[t].copy_(truncated)
        self._cursor = t + 1

    def flat(self) -> Dict[str, torch.Tensor]:
        """``(steps, num_envs, ...)`` -> ``(steps*num_envs, ...)`` views."""
        flat = self.steps * self.num_envs
        return {
            "obs": self.obs.reshape(flat, *self.obs.shape[2:]),
            "next_obs": self.next_obs.reshape(flat, *self.next_obs.shape[2:]),
            "action": self.actions.reshape(flat),
            "reward": self.rewards.reshape(flat),
            "logprob": self.logprobs.reshape(flat),
            "value": self.values.reshape(flat),
            "terminated": self.terminated.reshape(flat),
            "truncated": self.truncated.reshape(flat),
        }

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.obs,
                self.next_obs,
                self.actions,
                self.rewards,
                self.logprobs,
                self.values,
                self.terminated,
                self.truncated,
            )
        )
