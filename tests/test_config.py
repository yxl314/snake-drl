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
    # board/action/obs validation is snake-specific and lives in the adapter
    # (see tests/test_env.py::test_snake_validates_its_own_fields)
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


def test_zero_epoch_ppo_is_rejected():
    with pytest.raises(ValueError, match="epochs"):
        from_dict({"algorithm": {"type": "ppo", "epochs": 0}})


def test_jsonl_stays_valid_json_with_nan(tmp_path):
    import json

    from snake_drl.trainer.logger import MetricLogger

    with MetricLogger(tmp_path, console=False) as logger:
        logger.log(1, {"eval/return": float("nan"), "loss": 0.5})
    line = (tmp_path / "metrics.jsonl").read_text().splitlines()[0]
    record = json.loads(line, parse_constant=lambda c: pytest.fail(f"invalid constant {c}"))
    assert record["eval/return"] is None
    assert record["loss"] == 0.5


def test_custom_component_names_are_left_to_the_registries():
    """A user-registered encoder/buffer must be nameable from YAML; the config
    only checks shape, the registries decide which names exist."""
    from snake_drl.config import from_dict as build

    cfg = build({"network": {"encoder": {"type": "my_encoder"}}, "buffer": {"type": "my_buffer"}})
    assert cfg.network.encoder.type == "my_encoder"
    assert cfg.buffer.type == "my_buffer"
    # ...and an unknown name fails where the registry lives, with the list.
    from snake_drl.nn.encoders import build_encoder
    from snake_drl.spaces import ObsSpec

    with pytest.raises(ValueError, match="available: cnn"):
        build_encoder(ObsSpec((3, 8, 8)), cfg.network.encoder)


def test_override_must_be_key_value():
    from snake_drl.config import apply_overrides

    with pytest.raises(ValueError, match="KEY=VALUE"):
        apply_overrides({}, ["train.total_steps"])


def test_string_booleans_are_parsed_strictly():
    assert from_dict({"train": {"progress": "false"}}).train.progress is False
    assert from_dict({"train": {"progress": "true"}}).train.progress is True
    assert from_dict({"network": {"ortho_init": "no"}}).network.ortho_init is False
    with pytest.raises(ValueError, match="boolean"):
        from_dict({"train": {"progress": "maybe"}})


def test_unknown_key_error_names_the_key():
    with pytest.raises(ValueError, match="widthh"):
        from_dict({"env": {"widthh": 8}})


def test_algorithm_block_must_be_a_mapping():
    with pytest.raises(TypeError, match="algorithm block"):
        from_dict({"algorithm": None})


def test_ppo_rejects_dropout():
    with pytest.raises(ValueError, match="deterministic forward"):
        from_dict(
            {
                "algorithm": {"type": "ppo"},
                "network": {"encoder": {"dropout": 0.2}},
            }
        )
    # the DQN family is fine with it
    assert from_dict({"algorithm": {"type": "dqn"}, "network": {"encoder": {"dropout": 0.2}}})
