from __future__ import annotations

import numpy as np
import pytest
import torch
from snake_drl.algorithms import compute_gae, q_target
from snake_drl.config import from_dict
from snake_drl.device import configure_runtime, resolve_device
from snake_drl.trainer.factory import build_algorithm, build_evaluator

DEVICE = resolve_device("cpu")


def _algorithm(raw):
    cfg = from_dict(raw)
    configure_runtime(DEVICE, 4)
    return cfg, build_algorithm(cfg, configure_runtime(DEVICE, 4))


# --------------------------------------------------------------------------- #
# update maths
# --------------------------------------------------------------------------- #
def test_q_target_dqn_ddqn_and_termination():
    q_online_next = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
    q_target_next = torch.tensor([[4.0, 1.0], [1.0, 9.0]])
    reward = torch.tensor([1.0, -1.0])
    terminated = torch.tensor([False, True])

    dqn = q_target(q_target_next, q_online_next, reward, terminated, 0.9, double_q=False)
    ddqn = q_target(q_target_next, q_online_next, reward, terminated, 0.9, double_q=True)

    # env 0: DQN takes the target's max (4.0); DDQN follows the online argmax (1)
    assert torch.allclose(dqn, torch.tensor([4.6, -1.0]))
    assert torch.allclose(ddqn, torch.tensor([1.9, -1.0]))


def test_q_target_truncation_bootstraps():
    q_target_next = torch.tensor([[2.0]])
    q_online_next = torch.tensor([[2.0]])
    terminated = torch.tensor([False])  # truncated only -> still bootstraps
    out = q_target(q_target_next, q_online_next, torch.tensor([0.5]), terminated, 0.9, False)
    assert out.item() == pytest.approx(0.5 + 0.9 * 2.0)


def test_gae_matches_naive_reference():
    torch.manual_seed(0)
    steps, envs = 7, 4
    rewards = torch.randn(steps, envs)
    values = torch.randn(steps, envs)
    next_values = torch.randn(steps, envs)
    terminated = (torch.rand(steps, envs) > 0.75).float()

    adv, ret = compute_gae(rewards, values, next_values, terminated.bool(), 0.99, 0.95)

    reference = np.zeros((steps, envs))
    last = np.zeros(envs)
    for t in reversed(range(steps)):
        nonterm = 1.0 - terminated[t].numpy()
        delta = rewards[t].numpy() + 0.99 * next_values[t].numpy() * nonterm - values[t].numpy()
        last = delta + 0.99 * 0.95 * nonterm * last
        reference[t] = last
    assert np.allclose(adv.numpy(), reference, atol=1e-6)
    assert torch.allclose(ret, adv + values, atol=1e-6)


def test_dueling_head_centres_advantage():
    from snake_drl.nn.heads import DuelingQHead

    head = DuelingQHead(4, 3)
    feats = torch.randn(5, 4)
    q = head(feats)
    advantage = head.advantage(feats)
    assert torch.allclose(
        q - advantage + advantage.mean(dim=1, keepdim=True), head.value(feats), atol=1e-5
    )


# --------------------------------------------------------------------------- #
# learning on toy MDPs
# --------------------------------------------------------------------------- #
def _bandit_config(**overrides):
    """One-step terminal episodes: fast, deterministic and non-flaky."""
    raw = {
        "run": {"name": "bandit", "seed": 0, "device": "cpu", "output_dir": "/tmp/snake-drl-tests"},
        "env": {"type": "bandit", "num_envs": 16},
        "network": {"encoder": {"type": "mlp", "hidden_dims": [64], "feature_dim": 64}},
        "train": {"total_steps": 4000, "eval_episodes": 40, "eval_every": 10**9},
        "algorithm": {
            "type": "dqn",
            "lr": 3e-3,
            "batch_size": 64,
            "learning_starts": 128,
            "eps_decay_steps": 2000,
            "eps_end": 0.05,
            "target_update": 200,
            "train_freq": 1,
        },
    }
    for key, value in overrides.items():
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return raw


def test_dqn_family_learns_bandit():
    for variant in ("dqn", "ddqn", "d3qn"):
        cfg, algorithm = _algorithm(_bandit_config(**{"algorithm.type": variant}))
        while algorithm.env_steps < cfg.train.total_steps:
            algorithm.iterate()
        result = build_evaluator(cfg, DEVICE).run(algorithm.policy)
        assert result["eval/return"] > 0.9, (variant, result)
        algorithm.close()


def test_ppo_learns_bandit():
    raw = _bandit_config()
    raw["algorithm"] = {
        "type": "ppo",
        "lr": 3e-3,
        "n_steps": 32,
        "minibatch_size": 128,
        "epochs": 4,
    }
    raw["train"]["total_steps"] = 3000
    cfg, algorithm = _algorithm(raw)
    while algorithm.env_steps < cfg.train.total_steps:
        algorithm.iterate()
    result = build_evaluator(cfg, DEVICE).run(algorithm.policy)
    assert result["eval/return"] > 0.9, result
    algorithm.close()


def test_prioritized_replay_receives_td_errors():
    cfg, algorithm = _algorithm(_bandit_config(**{"train.total_steps": 1000}))
    step_metrics = []
    while algorithm.env_steps < 1000:
        step_metrics.append(algorithm.iterate())
    replay = algorithm.replay
    assert len(replay) > 0
    assert replay._tree.total > 0
    assert any(m.get("td_abs_mean", 0) > 0 for m in step_metrics if "td_abs_mean" in m)
    algorithm.close()


def test_iterations_and_updates_counters_are_consistent():
    raw = _bandit_config()
    raw["algorithm"] = {
        "type": "ppo",
        "lr": 3e-3,
        "n_steps": 16,
        "minibatch_size": 64,
        "epochs": 2,
    }
    cfg, algorithm = _algorithm(raw)
    reported = 0.0
    for _ in range(3):
        metrics = algorithm.iterate()
        reported += metrics["updates"]
    assert algorithm.iterations == 3
    assert algorithm.updates == int(reported)  # optimizer steps, not iterations
    assert algorithm.updates == 3 * 2 * 4  # epochs * minibatches (16*16/64)
    state = algorithm.state_dict()
    assert state["updates"] == algorithm.updates
    assert state["iterations"] == 3
    algorithm.close()


def test_q_target_requires_online_values_for_double_q():
    with pytest.raises(ValueError, match="online"):
        q_target(
            torch.zeros(2, 3),
            None,
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            0.99,
            double_q=True,
        )


def test_ppo_clips_gradients_on_every_minibatch(monkeypatch):
    """A one-shot parameters iterator would make clipping stop after step one."""
    seen: list[int] = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        batch = list(parameters)
        seen.append(len(batch))
        return original(batch, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    raw = _bandit_config()
    raw["algorithm"] = {
        "type": "ppo",
        "lr": 3e-3,
        "n_steps": 16,
        "minibatch_size": 64,
        "epochs": 2,
    }
    _, algorithm = _algorithm(raw)
    algorithm.iterate()
    algorithm.close()

    minibatches_per_epoch = (16 * 16) // 64
    assert len(seen) == 2 * minibatches_per_epoch
    assert all(count > 0 for count in seen), seen
