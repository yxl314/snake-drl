from __future__ import annotations

import json

import pytest
import torch

from snake_drl.config import from_dict
from snake_drl.device import configure_runtime, resolve_device
from snake_drl.trainer.factory import build_algorithm, build_trainer, load_policy
from snake_drl.trainer.trainer import Trainer

from tests.conftest import make_config

DEVICE = resolve_device("cpu")


def _config(tmp_path, **overrides):
    base = {
        "run.output_dir": str(tmp_path),
        "run.device": "cpu",
        "train.total_steps": 512,
        "train.log_every": 128,
        "train.eval_every": 256,
        "train.save_every": 256,
        "train.eval_episodes": 4,
        "train.progress": False,
        "algorithm.train_freq": 4,
    }
    base.update(overrides)
    return make_config(**base)


def test_trainer_runs_logs_evals_and_checkpoints(tmp_path):
    cfg = _config(tmp_path)
    trainer, run_dir = build_trainer(cfg)
    assert trainer.env_steps == 0
    trainer.run()

    assert trainer.env_steps >= cfg.train.total_steps
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "checkpoint_000000000512.pt").exists()

    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert records, "metrics.jsonl is empty"
    assert all("step" in r for r in records)
    assert any(k.startswith("eval/") for r in records for k in r)


def test_checkpoint_roundtrip_and_resume(tmp_path):
    cfg = _config(tmp_path, **{"train.eval_every": 10**9})
    trainer, run_dir = build_trainer(cfg)
    trainer.run()
    del trainer

    policy, loaded_cfg, payload = load_policy(run_dir / "latest.pt", "cpu")
    assert payload["env_steps"] == 512
    assert loaded_cfg.env.num_envs == cfg.env.num_envs
    assert loaded_cfg.network.encoder.type == cfg.network.encoder.type
    assert all(torch.isfinite(p).all() for p in policy.parameters())

    # resuming restores the policy weights and the counters
    resumed = build_algorithm(loaded_cfg, configure_runtime(DEVICE, 4))
    resumed.load_state_dict(payload["algorithm"])
    assert resumed.env_steps == 512
    for a, b in zip(policy.parameters(), resumed.policy.parameters()):
        assert torch.equal(a, b)
    resumed.close()


def test_trainer_respects_total_steps():
    cfg = from_dict(
        {
            "run": {"device": "cpu"},
            "env": {"type": "bandit", "num_envs": 4},
            "network": {"encoder": {"type": "mlp", "hidden_dims": [16], "feature_dim": 16}},
            "train": {"total_steps": 64, "progress": False},
            "algorithm": {"type": "ppo", "n_steps": 8, "minibatch_size": 16, "epochs": 1},
        }
    )
    algorithm = build_algorithm(cfg, configure_runtime(DEVICE, 4))
    trainer = Trainer(algorithm, total_steps=64)
    trainer.run()
    assert 64 <= algorithm.env_steps < 64 + 4 * 8
    algorithm.close()


def test_invalid_total_steps():
    with pytest.raises(ValueError):
        Trainer(algorithm=None, total_steps=0)  # type: ignore[arg-type]
