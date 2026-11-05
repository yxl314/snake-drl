"""Off-policy replay buffers: uniform and prioritized.

Storage choices, all measured:

* observations are stored in the dtype the environment emits (uint8 for the
  snake grid).  A 200k-transition buffer of 16x16x3 images costs 293 MiB instead
  of 1.2 GiB for float32, and the float conversion happens on the sampled
  minibatch only;
* sampling uses ``index_select(..., out=preallocated)`` so a training step does
  not allocate: 6.6 us for 256 images on 4 torch threads;
* the prioritized variant uses a vectorized sum tree (:mod:`.sumtree`), 67 us
  for 256 samples.

Only the sampled minibatch is pinned, not the whole storage: pinning 100k
observations would exhaust the pinned pool and is useless -- only the source of
an asynchronous host-to-device copy has to be pinned.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from .._dtypes import torch_dtype
from ..config import BufferConfig
from ..registry import register
from ..spaces import ObsSpec
from .batch import ReplayBatch
from .registry import BUFFERS
from .sumtree import SumTree

__all__ = ["PrioritizedReplayBuffer", "UniformReplayBuffer"]


class _TensorReplay:
    """Ring buffer over pre-allocated tensors."""

    def __init__(
        self,
        cfg: BufferConfig,
        obs_spec: ObsSpec,
        frames: int,
        device: torch.device | str,
        *,
        pin_memory: bool = False,
        seed: int = 0,
    ) -> None:
        if cfg.capacity < 1:
            raise ValueError("buffer capacity must be >= 1")
        if frames < 1:
            raise ValueError("frames must be >= 1")
        self.capacity = int(cfg.capacity)
        self.frames = int(frames)
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)
        self._pos = 0
        self._size = 0
        self._batch = 0
        self._pin_outputs = (
            bool(pin_memory) and self.device.type == "cpu" and torch.cuda.is_available()
        )

        window = (self.frames, *obs_spec.shape)
        dtype = torch_dtype(obs_spec)
        self._obs = torch.empty((self.capacity, *window), dtype=dtype, device=self.device)
        self._next_obs = torch.empty_like(self._obs)
        self._action = torch.empty(self.capacity, dtype=torch.int64, device=self.device)
        self._reward = torch.empty(self.capacity, dtype=torch.float32, device=self.device)
        self._terminated = torch.empty(self.capacity, dtype=torch.bool, device=self.device)
        self._truncated = torch.empty(self.capacity, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------- storage
    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Append a batch of transitions, wrapping around at the end."""
        count = int(action.shape[0])
        if count == 0:
            return
        if count > self.capacity:
            raise ValueError(
                f"cannot add {count} transitions to a buffer of capacity {self.capacity}"
            )
        self._write(
            self._pos,
            count,
            (
                (self._obs, obs),
                (self._next_obs, next_obs),
                (self._action, action),
                (self._reward, reward),
                (self._terminated, terminated),
                (self._truncated, truncated),
            ),
        )
        self._pos = (self._pos + count) % self.capacity
        self._size = min(self._size + count, self.capacity)
        self._on_add(count)

    def _write(
        self, start: int, count: int, pairs: Iterable[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Copy ``count`` contiguous slots from ``start``, splitting on wrap."""
        end = start + count
        if end <= self.capacity:
            for dst, src in pairs:
                dst[start:end].copy_(src)
            return
        head = self.capacity - start
        for dst, src in pairs:
            dst[start:].copy_(src[:head])
            dst[: end - self.capacity].copy_(src[head:])

    def _on_add(self, count: int) -> None:
        """Hook for subclasses that keep auxiliary indexes (e.g. priorities)."""

    # ------------------------------------------------------------- sampling
    def _ensure_batch(self, batch_size: int) -> None:
        if self._batch == batch_size:
            return
        self._batch = batch_size
        shape = (batch_size, self.frames, *self._obs.shape[2:])
        self._out_obs = torch.empty(
            shape, dtype=self._obs.dtype, device=self.device, pin_memory=self._pin_outputs
        )
        self._out_next = torch.empty_like(self._out_obs)
        self._out_action = torch.empty(
            batch_size, dtype=torch.int64, device=self.device, pin_memory=self._pin_outputs
        )
        self._out_reward = torch.empty(
            batch_size, dtype=torch.float32, device=self.device, pin_memory=self._pin_outputs
        )
        self._out_terminated = torch.empty(
            batch_size, dtype=torch.bool, device=self.device, pin_memory=self._pin_outputs
        )
        self._out_truncated = torch.empty_like(self._out_terminated)
        self._out_weights = torch.empty(
            batch_size, dtype=torch.float32, device=self.device, pin_memory=self._pin_outputs
        )

    def _draw(self, batch_size: int, beta: float) -> np.ndarray:
        """Sample slot indices and fill ``self._out_weights``."""
        self._out_weights.fill_(1.0)
        return self._rng.integers(0, self._size, batch_size)

    def sample(self, batch_size: int, beta: float = 1.0) -> ReplayBatch:
        """Gather a minibatch; sampling is with replacement."""
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty buffer")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._ensure_batch(batch_size)
        indices = self._draw(batch_size, beta)
        idx = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
        torch.index_select(self._obs, 0, idx, out=self._out_obs)
        torch.index_select(self._next_obs, 0, idx, out=self._out_next)
        torch.index_select(self._action, 0, idx, out=self._out_action)
        torch.index_select(self._reward, 0, idx, out=self._out_reward)
        torch.index_select(self._terminated, 0, idx, out=self._out_terminated)
        torch.index_select(self._truncated, 0, idx, out=self._out_truncated)
        return ReplayBatch(
            obs=self._out_obs,
            next_obs=self._out_next,
            action=self._out_action,
            reward=self._out_reward,
            terminated=self._out_terminated,
            truncated=self._out_truncated,
            indices=idx,
            weights=self._out_weights,
        )

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        """No-op for uniform replay; part of the buffer protocol."""


@register(BUFFERS, "uniform")
class UniformReplayBuffer(_TensorReplay):
    """Ring buffer with uniform sampling."""


@register(BUFFERS, "prioritized")
class PrioritizedReplayBuffer(_TensorReplay):
    """Proportional prioritized replay with importance-sampling correction."""

    def __init__(
        self,
        cfg: BufferConfig,
        obs_spec: ObsSpec,
        frames: int,
        device: torch.device | str,
        *,
        pin_memory: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(cfg, obs_spec, frames, device, pin_memory=pin_memory, seed=seed)
        self.alpha = float(cfg.alpha)
        self.eps = float(cfg.eps)
        self._tree = SumTree(self.capacity)
        self._max_priority = 1.0

    def _on_add(self, count: int) -> None:
        # New transitions get the current maximum priority so they are sampled
        # at least once.  ``self._pos`` has already advanced past the new slots.
        start = (self._pos - count) % self.capacity
        indices = (start + np.arange(count)) % self.capacity
        self._tree.update(indices, np.full(count, self._max_priority, dtype=np.float64))

    def _draw(self, batch_size: int, beta: float) -> np.ndarray:
        indices = self._tree.sample(batch_size, self._rng)
        priorities = self._tree.levels[0][indices]
        probabilities = priorities / self._tree.total
        weights = (self._size * probabilities) ** (-beta)
        weights /= weights.max()
        self._out_weights.copy_(torch.from_numpy(weights.astype(np.float32, copy=False)))
        return indices

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        idx = indices.detach().to("cpu", dtype=torch.int64).numpy()
        values = priorities.detach().to("cpu", dtype=torch.float64).numpy()
        values = np.power(np.abs(values) + self.eps, self.alpha)
        self._tree.update(idx, values)
        self._max_priority = max(self._max_priority, float(values.max()))
