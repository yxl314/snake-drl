"""Shared fixtures: everything runs on CPU, with tiny sizes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The package is installed editable, but tests should also work straight from a
# checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from snake_drl.config import from_dict  # noqa: E402
from snake_drl.device import configure_runtime, resolve_device  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _runtime():
    configure_runtime(resolve_device("cpu"), 4)


@pytest.fixture
def device():
    return resolve_device("cpu")


def make_config(**overrides):
    """A tiny config that builds in well under a second."""
    raw = {
        "run": {"name": "test", "seed": 0, "device": "cpu", "output_dir": "/tmp/snake-drl-tests"},
        "env": {
            "type": "snake",
            "num_envs": 8,
            "width": 8,
            "height": 8,
            "max_episode_steps": 50,
        },
        "network": {"encoder": {"type": "mlp", "hidden_dims": [32], "feature_dim": 32}},
        "train": {"total_steps": 64, "eval_episodes": 4, "eval_every": 10_000},
        "algorithm": {
            "type": "dqn",
            "batch_size": 32,
            "learning_starts": 16,
            "lr": 1e-3,
            "eps_decay_steps": 100,
        },
    }
    for key, value in overrides.items():
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return from_dict(raw)
