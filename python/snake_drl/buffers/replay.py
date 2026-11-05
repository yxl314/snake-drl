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
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from ..config import BufferConfig
from ..registry import register
from ..spaces import ObsSpec
from .base import BUFFERS, ReplayBatch
from .sumtree import SumTree

_TORCH_DTYPES = {"uint8": torch.uint8, "float32": torch.float32}


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
        self.capacity = int(cfg.capacity)
        self.frames = int(frames)
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)
        self._pos = 0
        self._size = 0
        window = (self.frames, *obs_spec.shape)
        dtype = _TORCH_DTYPES[obs_spec.dtype]
        pin = bool(pin_memory) and self.device.type == "cpu" and torch.cuda.is_available()

        self._obs = torch.empty((self.capacity, *window), dtype=dtype, device=self.device, pin_memory=pin)
        self._next_obs = torch.empty_like(self._obs)
        self._action = torch.empty(self.capacity, dtype=torch.int64, device=self.device)
        self._reward = torch.empty(self.capacity, dtype=torch.float32, device=self.device)
        self._terminated = torch.empty(self.capacity, dtype=torch.bool, device=self.device)
        self._truncated = torch.empty(self.capacity, dtype=torch.bool, device=self.device)
        self._batch = 0

    # ------------------------------------------------------------- storage
    def __len__(self) -> int:
        return self._size

    @property
    def position(self) -> int:
        return self._pos

    def _write(self, start: int, count: int, **arrays: torch.Tensor) -> None:
        """Write ``count`` contiguous slots from ``start``, wrapping if needed."""
        end = start + count
        cap = self.capacity
        if end <= cap:
            for name, value in arrays.items():
                getattr(self, name)[start:end].copy_(value)
            return
        head = cap - start
        for name, value in arrays.items():
            dst = getattr(self, name)
            dst[start:].copy_(value[:head])
            dst[: end - cap].copy_(value[head:])

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        count = int(action.shape[0])
        if count > self.capacity:
            raise ValueError("cannot add more transitions than the buffer capacity")
        self._write(
            self._pos,
            count,
            _obs=obs,
            _next_obs=next_obs,
            _action=action,
            _reward=reward,
            _terminated=terminated,
            _truncated=truncated,
        )
        self._pos = (self._pos + count) % self.capacity
        self._size = min(self._size + count, self.capacity)
        self._on_add(count)

    def _on_add(self, count: int) -> None:  # hook for prioritized
        pass

    # ------------------------------------------------------------- sampling
    def _ensure_batch(self, batch_size: int) -> None:
        if self._batch == batch_size:
            return
        self._batch = batch_size
        shape = (batch_size, self.frames, *self._obs.shape[2:])
        self._out_obs = torch.empty(shape, dtype=self._obs.dtype, device=self.device)
        self._out_next = torch.empty(shape, dtype=self._obs.dtype, device=self.device)
        self._out_action = torch.empty(batch_size, dtype=torch.int64, device=self.device)
        self._out_reward = torch.empty(batch_size, dtype=torch.float32, device=self.device)
        self._out_terminated = torch.empty(batch_size, dtype=torch.bool, device=self.device)
        self._out_truncated = torch.empty(batch_size, dtype=torch.bool, device=self.device)

    def _draw(self, batch_size: int, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        idx = self._rng.integers(0, self._size, batch_size)
        return idx, np.ones(batch_size, dtype=np.float32)

    def sample(self, batch_size: int, beta: float = 1.0) -> ReplayBatch:
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty buffer")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._ensure_batch(batch_size)
        idx_np, weights_np = self._draw(batch_size, beta)
        idx = torch.as_tensor(idx_np, dtype=torch.int64, device=self.device)
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
            weights=torch.as_tensor(weights_np, dtype=torch.float32, device=self.device),
        )

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        pass

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self._obs,
                self._next_obs,
                self._action,
                self._reward,
                self._terminated,
                self._truncated,
            )
        )


@register(BUFFERS, "uniform")
class UniformReplayBuffer(_TensorReplay):
    """Plain ring buffer with uniform sampling."""


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
        # New transitions get the current max priority so they are sampled at
        # least once; tree leaves are aligned with the ring slots.  ``self._pos``
        # has already been advanced past the freshly written slots.
        start = (self._pos - count) % self.capacity
        idx = (start + np.arange(count)) % self.capacity
        self._tree.update(idx, np.full(count, self._max_priority, dtype=np.float64))

    def _draw(self, batch_size: int, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        idx = self._tree.sample(batch_size, self._rng)
        priorities = self._tree.levels[0][idx]
        probs = priorities / self._tree.total
        weights = (self._size * probs) ** (-beta)
        weights = weights / weights.max()
        return idx, weights.astype(np.float32, copy=False)

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        idx_np = indices.detach().to("cpu", dtype=torch.int64).numpy()
        prio_np = priorities.detach().to("cpu", dtype=torch.float64).numpy()
        prio_np = np.power(np.abs(prio_np) + self.eps, self.alpha)
        self._tree.update(idx_np, prio_np)
        self._max_priority = max(self._max_priority, float(prio_np.max()))
