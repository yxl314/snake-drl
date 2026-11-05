"""Observation / action descriptions shared by every layer.

These are deliberately framework-light: they describe shapes and dtypes, not
tensors, so the environment layer never has to import torch and the network
layer never has to import the engine.

The whole library uses one tensor convention:

=============  =========================  ====================================
name           shape                      notes
=============  =========================  ====================================
``obs``        ``(B, T, *frame_shape)``   ``T`` = encoder receptive frames
``action``     ``(B,)`` int64
``reward``     ``(B,)`` float32
``terminated`` ``(B,)`` bool               no bootstrap
``truncated``  ``(B,)`` bool               bootstrap from the final observation
=============  =========================  ====================================
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce


@dataclass(frozen=True)
class ObsSpec:
    """A single-frame observation.

    Attributes:
        shape: per-frame shape -- ``(C, H, W)`` for images, ``(D,)`` for vectors.
        kind: ``"image"`` or ``"vector"``; encoders advertise what they accept.
        dtype: ``"uint8"`` or ``"float32"``, the dtype the environment emits.
        scale: multiply the float-cast observation by this before the encoder.
            Snake's grid planes are 0/1, so the default of 1.0 is right; an
            environment emitting 0..255 should use ``1/255``.
    """

    shape: tuple[int, ...]
    kind: str = "image"
    dtype: str = "uint8"
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in ("image", "vector"):
            raise ValueError(f"ObsSpec.kind must be 'image' or 'vector', got {self.kind!r}")
        if self.dtype not in ("uint8", "float32"):
            raise ValueError(f"ObsSpec.dtype must be 'uint8' or 'float32', got {self.dtype!r}")
        if not self.shape or any(int(s) <= 0 for s in self.shape):
            raise ValueError(f"ObsSpec.shape must be positive, got {self.shape}")
        if self.kind == "image" and len(self.shape) != 3:
            raise ValueError(f"image observations need (C,H,W), got {self.shape}")
        if self.kind == "vector" and len(self.shape) != 1:
            raise ValueError(f"vector observations need (D,), got {self.shape}")

    @property
    def frame_elems(self) -> int:
        return reduce(lambda a, b: a * b, self.shape, 1)

    def windowed(self, frames: int) -> tuple[int, ...]:
        return (int(frames), *self.shape)


@dataclass(frozen=True)
class ActionSpec:
    """Discrete action set (continuous control is out of scope for the snake)."""

    n: int
    kind: str = "discrete"

    def __post_init__(self) -> None:
        if self.kind != "discrete":
            raise ValueError("only discrete action spaces are supported")
        if int(self.n) <= 0:
            raise ValueError(f"ActionSpec.n must be positive, got {self.n}")


@dataclass(frozen=True)
class StepBatch:
    """One vectorized environment step, as device tensors.

    ``obs`` is the observation that *produced* ``action``; ``next_obs`` is the
    true successor observation.  For terminal and truncated steps ``next_obs`` is
    the final observation reported by the environment, not the reset state --
    that is what makes truncation bootstrapping correct.
    """

    obs: object  # torch.Tensor (B, T, *frame_shape)
    next_obs: object
    reward: object  # (B,) float32
    terminated: object  # (B,) bool
    truncated: object  # (B,) bool
