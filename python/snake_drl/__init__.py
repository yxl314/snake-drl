"""snake-drl: CPU batched snake engine with Python bindings."""

from ._snake_core import (
    AutoresetMode,
    ObsMode,
    VecConfig,
    VecSnake,
)

__all__ = ["AutoresetMode", "ObsMode", "VecConfig", "VecSnake", "__version__"]

__version__ = "1.0.0"
