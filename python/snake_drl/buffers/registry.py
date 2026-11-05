"""Replay-buffer registry (separate module to avoid import cycles)."""

from __future__ import annotations

from collections.abc import Callable

from ..registry import Registry

BUFFERS: Registry[Callable[..., object]] = Registry("replay buffer")
