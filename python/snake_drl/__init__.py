"""snake-drl: batched snake engine, human play mode and a modular DRL layer.

The core engine is imported eagerly (it is the reason the package exists); the
training layer -- and therefore torch -- is imported lazily so ``snake-play``
starts without paying for it.
"""

from ._snake_core import AutoresetMode, ObsMode, VecConfig, VecSnake

__version__ = "1.1.0"

__all__ = [
    "AutoresetMode",
    "ObsMode",
    "VecConfig",
    "VecSnake",
    "__version__",
    "ActionSpec",
    "BUFFERS",
    "ENCODERS",
    "ENVS",
    "Evaluator",
    "ObsSpec",
    "ProjectConfig",
    "Trainer",
    "build_policy",
    "build_trainer",
    "load_config",
    "load_policy",
    "make_env",
]

_LAZY = {
    "ActionSpec": ("snake_drl.spaces", "ActionSpec"),
    "ObsSpec": ("snake_drl.spaces", "ObsSpec"),
    "BUFFERS": ("snake_drl.buffers", "BUFFERS"),
    "ENCODERS": ("snake_drl.nn", "ENCODERS"),
    "ENVS": ("snake_drl.envs", "ENVS"),
    "Evaluator": ("snake_drl.trainer", "Evaluator"),
    "ProjectConfig": ("snake_drl.config", "ProjectConfig"),
    "Trainer": ("snake_drl.trainer", "Trainer"),
    "build_policy": ("snake_drl.nn", "build_policy"),
    "build_trainer": ("snake_drl.trainer", "build_trainer"),
    "load_config": ("snake_drl.config", "load"),
    "load_policy": ("snake_drl.trainer", "load_policy"),
    "make_env": ("snake_drl.envs", "make_env"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
