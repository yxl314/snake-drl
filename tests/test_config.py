from __future__ import annotations

import pytest
from snake_drl.config import DQNConfig, PPOConfig, from_dict, load, to_dict


def test_defaults_are_valid():
    cfg = from_dict({})
    assert isinstance(cfg.algorithm, DQNConfig)
    assert cfg.algorithm.type == "d3qn"
    assert cfg.algorithm.is_double and cfg.algorithm.is_dueling
    assert cfg.env.num_envs > 0


def test_algorithm_dispatch_and_presets():
    assert isinstance(from_dict({"algorithm": {"type": "ppo"}}).algorithm, PPOConfig)
    dqn = from_dict({"algorithm": {"type": "dqn"}}).algorithm
    ddqn = from_dict({"algorithm": {"type": "ddqn"}}).algorithm
    d3qn = from_dict({"algorithm": {"type": "d3qn"}}).algorithm
    assert not dqn.is_double and not dqn.is_dueling
    assert ddqn.is_double and not ddqn.is_dueling
    assert d3qn.is_double and d3qn.is_dueling


def test_explicit_flags_beat_preset():
    cfg = from_dict({"algorithm": {"type": "d3qn", "double_q": False, "dueling": False}})
    assert not cfg.algorithm.is_double and not cfg.algorithm.is_dueling


def test_unknown_key_is_rejected():
    with pytest.raises(Exception, match="typo_field"):
        from_dict({"network": {"typo_field": 1}})
    with pytest.raises(ValueError, match="unknown algorithm"):
        from_dict({"algorithm": {"type": "sac"}})


def test_yaml_and_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "algorithm:\n  type: ppo\n  lr: 0.001\nenv:\n  num_envs: 4\ntrain:\n  total_steps: 100\n",
        encoding="utf-8",
    )
    cfg = load(str(path), ["train.total_steps=250", "network.encoder.frames=1"])
    assert cfg.train.total_steps == 250
    assert cfg.algorithm.lr == pytest.approx(1e-3)
    assert cfg.env.num_envs == 4


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError):
        from_dict({"env": {"num_envs": 0}})
    with pytest.raises(ValueError):
        from_dict({"env": {"obs_mode": "bitboard"}})
    with pytest.raises(ValueError):
        from_dict({"network": {"encoder": {"type": "gru", "frames": 1}}})
    with pytest.raises(ValueError):
        from_dict({"algorithm": {"type": "ppo", "lr_schedule": "cosine"}})


def test_roundtrip_dict():
    cfg = from_dict({"env": {"num_envs": 3}})
    again = from_dict(to_dict(cfg))
    assert again.env.num_envs == 3
    assert isinstance(again.algorithm, DQNConfig)


def test_policy_family_follows_the_algorithm():
    assert from_dict({"algorithm": {"type": "ppo"}}).policy_family == "actor_critic"
    assert from_dict({"algorithm": {"type": "d3qn"}}).policy_family == "q"
    assert from_dict({"algorithm": {"type": "ppo"}}).is_on_policy is True


def test_apply_overrides_precedence():
    from snake_drl.config import apply_overrides

    raw = {"algorithm": {"type": "ppo", "lr": 0.001}, "env": {"num_envs": 4}}
    merged = apply_overrides(raw, ["algorithm.type=d3qn", "env.num_envs=8"])
    assert merged["algorithm"]["type"] == "d3qn"
    assert merged["algorithm"]["lr"] == 0.001  # untouched keys survive
    assert merged["env"]["num_envs"] == 8
    assert apply_overrides(raw, []) is raw


def test_removed_knobs_are_rejected():
    # ``grad_clip`` was folded into ``max_grad_norm == 0``; strict parsing keeps
    # old configs from silently half-applying.
    with pytest.raises(Exception, match="grad_clip"):
        from_dict({"algorithm": {"type": "dqn", "grad_clip": True}})
