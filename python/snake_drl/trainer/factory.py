"""Composition root: the only place that knows how the pieces fit together.

Everything is constructed explicitly and passed as constructor arguments.
Algorithms never instantiate encoders or buffers themselves, which is what keeps
the swap-a-component story honest (and testable with fakes).
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
from omegaconf import OmegaConf

from ..algorithms.base import Algorithm
from ..algorithms.dqn import DQN
from ..algorithms.exploration import LinearSchedule
from ..algorithms.ppo import PPO
from ..buffers import RolloutBuffer, build_replay
from ..config import DQNConfig, PPOConfig, ProjectConfig, apply_overrides, from_dict, to_dict
from ..device import Runtime, configure_runtime, resolve_device
from ..envs import VectorEnv, make_env
from ..nn.build import build_policy
from ..nn.policies import ActorCriticPolicy, QPolicy
from ..runner import VectorRunner
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
from .trainer import Trainer

__all__ = ["build_algorithm", "build_evaluator", "build_trainer", "load_policy"]


def _build_policy(
    cfg: ProjectConfig, obs_spec: ObsSpec, action_spec: ActionSpec, device: torch.device
) -> ActorCriticPolicy | QPolicy:
    dueling = bool(getattr(cfg.algorithm, "is_dueling", False))
    policy = build_policy(
        cfg.network,
        obs_spec,
        action_spec,
        family=cfg.policy_family,
        device=device,
        dueling=dueling,
    )
    assert isinstance(policy, (ActorCriticPolicy, QPolicy))
    return policy


def _replay_device(cfg: ProjectConfig, compute: torch.device) -> torch.device:
    if cfg.buffer.device == "auto":
        return compute
    return resolve_device(cfg.buffer.device)


def build_algorithm(
    cfg: ProjectConfig,
    runtime: Runtime,
    *,
    env: VectorEnv | None = None,
    seed: int | None = None,
    total_steps: int | None = None,
) -> Algorithm:
    """Build environment, policy, buffer and optimizer, then the algorithm.

    This is a composition root, so it also seeds the process: constructing an
    algorithm is the moment a reproducible run starts.
    """
    seed_all(cfg.run.seed if seed is None else seed)
    env = env or make_env(cfg.env, seed=cfg.env.seed)
    policy = _build_policy(cfg, env.obs_spec, env.action_spec, runtime.device)
    runner = VectorRunner(env, policy.frames, runtime.device, pin_memory=cfg.run.pin_memory)
    horizon = cfg.train.total_steps if total_steps is None else total_steps

    if isinstance(cfg.algorithm, PPOConfig):
        ppo_cfg = cfg.algorithm
        rollout = RolloutBuffer(
            ppo_cfg.n_steps, env.num_envs, env.obs_spec, policy.frames, runtime.device
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_cfg.lr, eps=ppo_cfg.adam_eps)
        return PPO(
            ppo_cfg,
            cast(ActorCriticPolicy, policy),
            runner,
            rollout,
            optimizer,
            runtime.device,
            total_steps=horizon,
        )

    if isinstance(cfg.algorithm, DQNConfig):
        dqn_cfg = cfg.algorithm
        replay = build_replay(
            cfg.buffer,
            env.obs_spec,
            policy.frames,
            _replay_device(cfg, runtime.device),
            pin_memory=cfg.run.pin_memory,
            seed=derive_seed(cfg.run.seed, 7),
        )
        q_policy = cast(QPolicy, policy)
        target = copy.deepcopy(q_policy).to(runtime.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=dqn_cfg.lr, eps=dqn_cfg.adam_eps)
        return DQN(
            dqn_cfg,
            q_policy,
            target,
            runner,
            replay,
            optimizer,
            runtime.device,
            epsilon=LinearSchedule(dqn_cfg.eps_start, dqn_cfg.eps_end, dqn_cfg.eps_decay_steps),
            beta=LinearSchedule(cfg.buffer.beta0, 1.0, cfg.buffer.beta_steps),
            total_steps=horizon,
        )

    raise TypeError(  # pragma: no cover - ProjectConfig validates this
        f"unsupported algorithm config {type(cfg.algorithm).__name__}"
    )


def _eval_env_factory(cfg: ProjectConfig) -> Callable[[], VectorEnv]:
    """Evaluation gets its own environment batch and its own seed."""
    eval_cfg = replace(
        cfg.env,
        num_envs=min(16, cfg.env.num_envs),
        seed=derive_seed(cfg.run.seed, 1),
    )
    return lambda: make_env(eval_cfg, seed=eval_cfg.seed)


def build_evaluator(cfg: ProjectConfig, device: torch.device) -> Evaluator:
    return Evaluator(
        _eval_env_factory(cfg),
        device,
        episodes=cfg.train.eval_episodes,
        max_steps=cfg.train.eval_max_steps,
    )


def _write_config(run_dir: Path, cfg: ProjectConfig) -> None:
    (run_dir / "config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(to_dict(cfg))), encoding="utf-8"
    )


def build_trainer(
    cfg: ProjectConfig, *, device: torch.device | None = None
) -> tuple[Trainer, Path]:
    """Assemble a whole run. Returns the trainer and its output directory."""
    compute = device or resolve_device(cfg.run.device)
    runtime = configure_runtime(compute, cfg.run.torch_threads)
    algorithm = build_algorithm(cfg, runtime)

    run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_config(run_dir, cfg)

    logger = MetricLogger(run_dir)
    callbacks = [
        LoggingCallback(logger, cfg.train.log_every),
        EvalCallback(build_evaluator(cfg, runtime.device), cfg.train.eval_every, logger),
        CheckpointCallback(run_dir, cfg.train.save_every, cfg, cfg.train.keep_last),
        CloseLoggerCallback(logger),
    ]
    if cfg.train.progress:
        callbacks.append(ProgressCallback())
    return Trainer(algorithm, cfg.train.total_steps, callbacks), run_dir


def load_policy(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    *,
    overrides: list[str] | None = None,
) -> tuple[torch.nn.Module, ProjectConfig, dict]:
    """Rebuild a policy from a checkpoint (used by ``snake-drl eval``).

    ``overrides`` are applied on top of the checkpoint's own config, so an
    evaluation can change, say, ``env.num_envs`` without inventing a new config
    that disagrees with the saved weights.
    """
    from .checkpoint import load_checkpoint

    device = torch.device(device)
    payload = load_checkpoint(checkpoint_path, map_location=device)
    raw = apply_overrides(dict(payload["config"]), list(overrides or []))
    cfg = from_dict(raw)
    env = make_env(replace(cfg.env, num_envs=1), seed=cfg.env.seed)
    try:
        policy = _build_policy(cfg, env.obs_spec, env.action_spec, device)
    finally:
        env.close()
    policy.load_state_dict(payload["algorithm"]["policy"])
    policy.eval()
    return policy, cfg, payload
