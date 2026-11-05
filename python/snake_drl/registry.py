"""A tiny name -> factory registry.

Components (encoders, buffers, algorithms, environments) register themselves
here so a YAML file can name them.  Factories are plain callables taking keyword
arguments; nothing is instantiated implicitly, which keeps the wiring explicit
and testable (no ``_target_`` magic, no global state beyond the registry itself).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

__all__ = ["Registry", "register"]

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, key: str, item: T | None = None) -> Any:
        """Register ``item`` under ``key``; usable directly or as a decorator."""

        def _decorator(obj: T) -> T:
            if key in self._items:
                raise KeyError(f"{self.name} {key!r} is already registered")
            self._items[key] = obj
            return obj

        if item is not None:
            return _decorator(item)
        return _decorator

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"unknown {self.name} {key!r}; available: {available}") from None

    def create(self, key: str, **kwargs: Any) -> Any:
        return self.get(key)(**kwargs)  # type: ignore[operator]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)


def register(registry: Registry[T], key: str) -> Callable[[T], T]:
    """Decorator form, e.g. ``@register(ENCODERS, "cnn")``."""

    def _decorator(obj: T) -> T:
        registry.register(key, obj)
        return obj

    return _decorator
