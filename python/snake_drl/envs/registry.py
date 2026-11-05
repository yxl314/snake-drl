"""Environment registry.

A separate module (rather than living next to the base class) so concrete
environments can import it without importing the package ``__init__`` and
creating an import cycle.
"""

from __future__ import annotations

from collections.abc import Callable

from ..registry import Registry
from .base import VectorEnv

ENVS: Registry[Callable[..., VectorEnv]] = Registry("environment")
