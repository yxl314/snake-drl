"""Shapes and dtypes of the data every layer exchanges.

This module is intentionally framework-light: it describes observation and
action spaces with plain tuples and strings, so the environment layer never has
to import torch and the network layer never has to import the engine.

One tensor convention is shared by the whole training stack:

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

__all__ = ["ActionSpec", "ObsSpec"]

_KINDS = ("image", "vector")
_DTYPES = ("uint8", "float32")


@dataclass(frozen=True)
class ObsSpec:
    """One frame of an observation.

    Attributes:
        shape: per-frame shape -- ``(C, H, W)`` for images, ``(D,)`` for vectors.
        kind: ``"image"`` or ``"vector"``; encoders advertise what they accept.
        dtype: ``"uint8"`` or ``"float32"``, the dtype the environment emits.
            It is kept as a string so this module does not import numpy or torch.
        scale: multiply the float-cast observation by this before the encoder.
            Snake's grid planes are 0/1, so 1.0 is right; an environment that
            emits 0..255 should use ``1/255``.
    """

    shape: tuple[int, ...]
    kind: str = "image"
    dtype: str = "uint8"
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"ObsSpec.kind must be one of {_KINDS}, got {self.kind!r}")
        if self.dtype not in _DTYPES:
            raise ValueError(f"ObsSpec.dtype must be one of {_DTYPES}, got {self.dtype!r}")
        if not self.shape or any(int(s) <= 0 for s in self.shape):
            raise ValueError(f"ObsSpec.shape must be non-empty and positive, got {self.shape}")
        if self.kind == "image" and len(self.shape) != 3:
            raise ValueError(f"image observations need (C, H, W), got {self.shape}")
        if self.kind == "vector" and len(self.shape) != 1:
            raise ValueError(f"vector observations need (D,), got {self.shape}")
        if self.scale <= 0.0:
            raise ValueError(f"ObsSpec.scale must be positive, got {self.scale}")

    @property
    def frame_elems(self) -> int:
        """Number of scalar values in one frame."""
        return reduce(lambda a, b: a * b, self.shape, 1)


@dataclass(frozen=True)
class ActionSpec:
    """A discrete action set (continuous control is out of scope for the snake)."""

    n: int
    kind: str = "discrete"

    def __post_init__(self) -> None:
        if self.kind != "discrete":
            raise ValueError("only discrete action spaces are supported")
        if int(self.n) <= 0:
            raise ValueError(f"ActionSpec.n must be positive, got {self.n}")
