"""Command line entry point.

    snake-drl train -c configs/d3qn_cnn.yaml --set train.total_steps=10000
    snake-drl eval  -m runs/<run>/latest.pt --episodes 20
    snake-drl play
    snake-drl bench -c configs/d3qn_cnn.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch

from .config import load
from .device import configure_runtime, resolve_device


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", type=str, default=None, help="YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set train.total_steps=10000",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="snake-drl", description="Modular DRL for the snake engine.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train an agent")
    _add_common(train)

    ev = sub.add_parser("eval", help="evaluate a checkpoint")
    ev.add_argument("-m", "--model", type=str, required=True, help="checkpoint path")
    ev.add_argument("-c", "--config", type=str, default=None, help="config override")
    ev.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ev.add_argument("--episodes", type=int, default=None)
    ev.add_argument("--device", type=str, default=None)

    play = sub.add_parser("play", help="play the game yourself (PyQt6)")
    play.add_argument("--width", type=int, default=16)
    play.add_argument("--height", type=int, default=16)
    play.add_argument("--seed", type=int, default=None)
    play.add_argument("--speed", type=int, default=4)

    bench = sub.add_parser("bench", help="measure env + policy throughput")
    _add_common(bench)
    bench.add_argument("--steps", type=int, default=200)
    bench.add_argument("--warmup", type=int, default=20)
    return parser


def cmd_train(args: argparse.Namespace) -> int:
    from .trainer import build_trainer, build_algorithm  # noqa: F401 (public API)

    cfg = load(args.config, args.overrides)
    trainer, run_dir = build_trainer(cfg)
    policy = trainer.algorithm.policy
    params = sum(p.numel() for p in policy.parameters())
    print(
        f"run dir   : {run_dir}\n"
        f"device    : {trainer.algorithm.device}\n"
        f"algorithm : {type(cfg.algorithm).__name__} ({cfg.algorithm.type})\n"
        f"encoder   : {cfg.network.encoder.type} frames={cfg.network.encoder.frames}\n"
        f"policy    : {type(policy).__name__} ({params/1e3:.1f}k params)\n"
        f"env       : {cfg.env.num_envs} x {cfg.env.width}x{cfg.env.height}\n"
        f"steps     : {cfg.train.total_steps}",
        flush=True,
    )
    try:
        trainer.run()
    finally:
        trainer.algorithm.close()
    print(f"done: {run_dir}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .config import from_dict
    from .trainer import build_evaluator, load_policy

    device = resolve_device(args.device or "auto")
    configure_runtime(device)
    policy, cfg, payload = load_policy(args.model, device)
    if args.config or args.overrides:
        cfg = load(args.config, args.overrides)
    if args.episodes:
        cfg.train.eval_episodes = args.episodes
    evaluator = build_evaluator(cfg, device)
    t0 = time.perf_counter()
    metrics = evaluator.run(policy)
    elapsed = time.perf_counter() - t0
    print(f"checkpoint: {args.model} (trained {payload.get('env_steps', 0)} env steps)")
    for key, value in metrics.items():
        print(f"  {key:18s} {value:.4f}")
    print(f"  elapsed            {elapsed:.1f}s")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    from .play.app import main as play_main

    argv: List[str] = ["--width", str(args.width), "--height", str(args.height), "--speed", str(args.speed)]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    return play_main(argv)


def cmd_bench(args: argparse.Namespace) -> int:
    from .envs import make_env
    from .nn.build import build_policy
    from .runner import VectorRunner
    from .trainer.factory import policy_family

    cfg = load(args.config, args.overrides)
    device = resolve_device(cfg.run.device)
    runtime = configure_runtime(device, cfg.run.torch_threads)
    env = make_env(cfg.env, seed=cfg.env.seed)
    policy = build_policy(
        cfg.network,
        env.obs_spec,
        env.action_spec,
        family=policy_family(cfg),
        device=device,
        dueling=bool(getattr(cfg.algorithm, "is_dueling", False)),
    )
    runner = VectorRunner(env, policy.frames, device, pin_memory=cfg.run.pin_memory)

    @torch.no_grad()
    def step_once(obs: torch.Tensor) -> torch.Tensor:
        if policy_family(cfg) == "actor_critic":
            action = policy.act(obs).action
        else:
            action = policy.act(obs, epsilon=0.05)
        runner.step(action)
        return runner.obs

    obs = runner.reset()
    for _ in range(args.warmup):
        obs = step_once(obs)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        obs = step_once(obs)
    dt = time.perf_counter() - t0
    total = args.steps * env.num_envs
    print(
        f"threads={runtime.torch_threads} device={device} encoder={cfg.network.encoder.type} "
        f"frames={policy.frames}\n"
        f"{total} env steps in {dt:.2f}s -> {total/dt:,.0f} env-step/s "
        f"({dt/args.steps*1e3:.2f} ms per vectorized step, {env.num_envs} envs)"
    )
    runner.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "train": cmd_train,
        "eval": cmd_eval,
        "play": cmd_play,
        "bench": cmd_bench,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
