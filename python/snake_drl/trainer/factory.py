"""Composition root: the only place that knows how the pieces fit together.

Everything below is constructed explicitly and passed as constructor arguments.
Algorithms never instantiate encoders or buffers themselves, which is what keeps
the swap-a-component story honest (and testable with fakes).
"""

from __future__ import annotations

import copy
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from omegaconf import OmegaConf

from ..algorithms.base import Algorithm
from ..algorithms.dqn import DQN
from ..algorithms.exploration import LinearSchedule
from ..algorithms.ppo import PPO
from ..buffers import RolloutBuffer, build_replay
from ..config import DQNConfig, PPOConfig, ProjectConfig, RunConfig, to_dict
from ..device import Runtime, configure_runtime, resolve_device
from ..envs import VectorEnv, make_env
from ..nn.build import build_policy
from ..seeding import derive_seed, seed_all
from ..spaces import ActionSpec, ObsSpec
from .callbacks import (
    CheckpointCallback,
    CloseLoggerCallback,
    EvalCallback,
    LoggingCallback,
    ProgressCallback,
)
from .evaluator import Evaluator
from .logger import MetricLogger
from ..runner import VectorRunner
from .trainer import Trainer


def policy_family(cfg: ProjectConfig) -> str:
    return "actor_critic" if isinstance(cfg.algorithm, PPOConfig) else "q"


def build_policy_for(
    cfg: ProjectConfig, obs_spec: ObsSpec, action_spec: ActionSpec, device: torch.device
) -> torch.nn.Module:
    dueling = bool(getattr(cfg.algorithm, "is_dueling", False))
    return build_policy(
        cfg.network,
        obs_spec,
        action_spec,
        family=policy_family(cfg),
        device=device,
        dueling=dueling,
    )


def buffer_device(cfg: ProjectConfig, compute: torch.device) -> torch.device:
    if cfg.buffer.device == "auto":
        return compute
    return resolve_device(cfg.buffer.device)


def build_algorithm(
    cfg: ProjectConfig,
    runtime: Runtime,
    *,
    env: Optional[VectorEnv] = None,
    seed: Optional[int] = None,
) -> Algorithm:
    """Build the environment, policy, buffer and optimizer, then the algorithm."""
    seed_all(cfg.run.seed if seed is None else seed)
    env = env or make_env(cfg.env, seed=cfg.env.seed)
    policy = build_policy_for(cfg, env.obs_spec, env.action_spec, runtime.device)
    runner = VectorRunner(env, policy.frames, runtime.device, pin_memory=cfg.run.pin_memory)

    if isinstance(cfg.algorithm, PPOConfig):
        algo_cfg = cfg.algorithm
        rollout = RolloutBuffer(
            algo_cfg.n_steps, env.num_envs, env.obs_spec, policy.frames, runtime.device
        )
        optimizer = torch.optim.Adam(
            policy.parameters(), lr=algo_cfg.lr, eps=algo_cfg.adam_eps
        )
        algorithm: Algorithm = PPO(algo_cfg, policy, runner, rollout, optimizer, runtime.device)
    elif isinstance(cfg.algorithm, DQNConfig):
        algo_cfg = cfg.algorithm
        replay = build_replay(
            cfg.buffer,
            env.obs_spec,
            policy.frames,
            buffer_device(cfg, runtime.device),
            pin_memory=cfg.run.pin_memory,
            seed=derive_seed(cfg.run.seed, 7),
        )
        target = copy.deepcopy(policy).to(runtime.device)
        optimizer = torch.optim.Adam(
            policy.parameters(), lr=algo_cfg.lr, eps=algo_cfg.adam_eps
        )
        algorithm = DQN(
            algo_cfg,
            policy,
            target,
            runner,
            replay,
            optimizer,
            runtime.device,
            epsilon=LinearSchedule(algo_cfg.eps_start, algo_cfg.eps_end, algo_cfg.eps_decay_steps),
            beta=LinearSchedule(cfg.buffer.beta0, 1.0, cfg.buffer.beta_steps),
        )
    else:  # pragma: no cover - ProjectConfig validates this
        raise TypeError(f"unsupported algorithm config {type(cfg.algorithm).__name__}")

    algorithm.total_steps = cfg.train.total_steps
    return algorithm


def make_eval_env_factory(cfg: ProjectConfig) -> Callable[[], VectorEnv]:
    eval_cfg = replace(
        cfg.env,
        num_envs=min(16, cfg.env.num_envs),
        seed=derive_seed(cfg.run.seed, 1),
    )

    def _make() -> VectorEnv:
        return make_env(eval_cfg, seed=eval_cfg.seed)

    return _make


def build_evaluator(cfg: ProjectConfig, device: torch.device) -> Evaluator:
    return Evaluator(
        make_eval_env_factory(cfg),
        device,
        episodes=cfg.train.eval_episodes,
        max_steps=cfg.train.eval_max_steps,
    )


def build_trainer(cfg: ProjectConfig, *, device: Optional[torch.device] = None) -> Tuple[Trainer, Path]:
    """Assemble the whole run. Returns the trainer and its output directory."""
    compute = device or resolve_device(cfg.run.device)
    runtime = configure_runtime(compute, cfg.run.torch_threads)
    algorithm = build_algorithm(cfg, runtime)

    run_dir = Path(cfg.run.output_dir) / (
        f"{cfg.run.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(to_dict(cfg))), encoding="utf-8"
    )

    logger = MetricLogger(run_dir)
    evaluator = build_evaluator(cfg, runtime.device)
    callbacks = [
        LoggingCallback(logger, cfg.train.log_every),
        EvalCallback(evaluator, cfg.train.eval_every, logger),
        CheckpointCallback(run_dir, cfg.train.save_every, cfg, cfg.train.keep_last),
        CloseLoggerCallback(logger),
    ]
    if cfg.train.progress:
        callbacks.append(ProgressCallback())
    return Trainer(algorithm, cfg.train.total_steps, callbacks), run_dir


def load_policy(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> Tuple[torch.nn.Module, ProjectConfig, dict]:
    """Rebuild a policy from a checkpoint (used by ``snake-drl eval``)."""
    from .checkpoint import load_checkpoint

    device = torch.device(device)
    payload = load_checkpoint(checkpoint_path, map_location=device)
    cfg = _config_from_payload(payload)
    env = make_env(replace(cfg.env, num_envs=1), seed=cfg.env.seed)
    policy = build_policy_for(cfg, env.obs_spec, env.action_spec, device)
    env.close()
    policy.load_state_dict(payload["algorithm"]["policy"])
    policy.eval()
    return policy, cfg, payload


def _config_from_payload(payload: dict) -> ProjectConfig:
    from ..config import from_dict

    return from_dict(payload["config"])
