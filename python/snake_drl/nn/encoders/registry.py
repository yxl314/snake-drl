"""Encoder registry.

Kept in its own module so encoder implementations can register themselves
without importing the package ``__init__`` (which would be a cycle).
"""

from __future__ import annotations

from ...registry import Registry

ENCODERS: Registry[type] = Registry("encoder")
