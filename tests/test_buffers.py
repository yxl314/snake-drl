from __future__ import annotations

import numpy as np
import pytest
import torch

from snake_drl.buffers import (
    PrioritizedReplayBuffer,
    RolloutBuffer,
    SumTree,
    UniformReplayBuffer,
)
from snake_drl.config import BufferConfig
from snake_drl.spaces import ObsSpec

IMAGE = ObsSpec((2, 3, 3))
VECTOR = ObsSpec((4,), kind="vector", dtype="float32")


# --------------------------------------------------------------------------- #
# sum tree
# --------------------------------------------------------------------------- #
def test_sum_tree_matches_bruteforce_distribution():
    cap = 64
    tree = SumTree(cap)
    rng = np.random.default_rng(0)
    priorities = np.linspace(0.1, 10.0, cap)
    tree.update(np.arange(cap), priorities)
    assert tree.total == pytest.approx(priorities.sum())

    samples = tree.sample(200_000, rng)
    freq = np.bincount(samples, minlength=cap) / 200_000
    expected = priorities / priorities.sum()
    assert np.max(np.abs(freq - expected)) < 0.01
    # deterministic for a fixed rng
    assert np.array_equal(tree.sample(10, np.random.default_rng(3)),
                          tree.sample(10, np.random.default_rng(3)))


def test_sum_tree_duplicate_indices_stay_consistent():
    """Sampling can repeat a slot; the update must not double count it."""
    tree = SumTree(8)
    tree.update(np.arange(8), np.ones(8))
    tree.update(np.array([2, 2, 2]), np.array([0.5, 0.5, 0.5]))
    assert tree.total == pytest.approx(7.5)
    tree.update(np.array([2, 2]), np.array([1.0, 3.0]))
    # one write per slot is applied; the invariant that matters is consistency
    assert tree.total == pytest.approx(float(tree.levels[0].sum()))
    assert tree.total in (pytest.approx(8.0), pytest.approx(10.0))


def test_sum_tree_rejects_empty():
    with pytest.raises(RuntimeError):
        SumTree(4).sample(3, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# replay buffers
# --------------------------------------------------------------------------- #
def _batch(n: int, spec: ObsSpec, value: int = 0):
    shape = (n, 1, *spec.shape)
    obs = torch.full(shape, value, dtype=torch.uint8 if spec.dtype == "uint8" else torch.float32)
    return dict(
        obs=obs,
        next_obs=obs.clone(),
        action=torch.zeros(n, dtype=torch.long),
        reward=torch.zeros(n),
        terminated=torch.zeros(n, dtype=torch.bool),
        truncated=torch.zeros(n, dtype=torch.bool),
    )


def test_uniform_replay_wraps_and_samples_shapes():
    buf = UniformReplayBuffer(BufferConfig(type="uniform", capacity=10), IMAGE, 1, "cpu")
    for i in range(12):
        buf.add(**_batch(1, IMAGE, value=i % 256))
    assert len(buf) == 10  # capped
    batch = buf.sample(4)
    assert batch.obs.shape == (4, 1, 2, 3, 3)
    assert batch.action.shape == (4,)
    assert torch.all(batch.weights == 1.0)
    assert batch.obs.dtype == torch.uint8


def test_replay_batched_add_matches_single_add():
    buf = UniformReplayBuffer(BufferConfig(type="uniform", capacity=16), IMAGE, 1, "cpu")
    buf.add(**_batch(8, IMAGE, value=1))
    buf.add(**_batch(4, IMAGE, value=2))
    assert len(buf) == 12
    sampled = buf.sample(12)
    values = sampled.obs.flatten(1)[:, 0]
    assert set(values.tolist()) == {1, 2}


def test_prioritized_replay_weights_and_new_samples():
    cfg = BufferConfig(type="prioritized", capacity=64, alpha=1.0, beta0=1.0)
    buf = PrioritizedReplayBuffer(cfg, VECTOR, 1, "cpu", seed=0)
    buf.add(**_batch(64, VECTOR))
    assert buf._max_priority == pytest.approx(1.0)
    first = buf.sample(64)
    assert torch.allclose(first.weights, torch.ones(64))

    # make every slot equally unlikely; the tree must follow
    buf.update_priorities(torch.arange(64), torch.full((64,), 0.01))
    tree_probs = buf._tree.levels[0][:64] / buf._tree.total
    assert np.allclose(tree_probs, 0.01 / (0.01 * 64))

    # a slot updated to a much larger priority must dominate sampling
    buf.update_priorities(torch.tensor([3]), torch.tensor([10.0]))
    sampled = buf.sample(4_000)
    assert (sampled.indices == 3).float().mean() > 0.9


def test_prioritized_is_weights_normalised():
    cfg = BufferConfig(type="prioritized", capacity=32, alpha=0.6, beta0=0.6)
    buf = PrioritizedReplayBuffer(cfg, VECTOR, 1, "cpu", seed=1)
    buf.add(**_batch(32, VECTOR))
    rng = np.random.default_rng(0)
    buf.update_priorities(torch.arange(32), torch.from_numpy(rng.random(32) * 5))
    batch = buf.sample(16, beta=0.6)
    assert batch.weights.min() > 0
    assert batch.weights.max() <= 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# rollout buffer
# --------------------------------------------------------------------------- #
def test_rollout_buffer_shapes_and_flatten():
    buf = RolloutBuffer(steps=4, num_envs=3, obs_spec=IMAGE, frames=2, device="cpu")
    for _ in range(4):
        batch = _batch(3, IMAGE)
        buf.add(batch["obs"].repeat(1, 2, 1, 1, 1), batch["next_obs"].repeat(1, 2, 1, 1, 1),
                batch["action"], batch["reward"], torch.zeros(3), torch.zeros(3),
                batch["terminated"], batch["truncated"])
    assert buf.cursor == 4
    flat = buf.flat()
    assert flat["obs"].shape == (12, 2, 2, 3, 3)
    assert flat["action"].shape == (12,)
    with pytest.raises(RuntimeError, match="full"):
        batch = _batch(3, IMAGE)
        buf.add(batch["obs"].repeat(1, 2, 1, 1, 1), batch["next_obs"].repeat(1, 2, 1, 1, 1),
                batch["action"], batch["reward"], torch.zeros(3), torch.zeros(3),
                batch["terminated"], batch["truncated"])
    buf.reset()
    assert buf.cursor == 0
