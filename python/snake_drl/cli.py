"""Command line entry point.

snake-drl train -c configs/d3qn_cnn.yaml --set train.total_steps=10000
snake-drl eval  -m runs/<run>/latest.pt --episodes 20
snake-drl play
snake-drl bench -c configs/dqn_mlp.yaml
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from .config import load
from .device import configure_runtime, resolve_device

__all__ = ["main"]

_SET_HELP = "override a config value, e.g. --set train.total_steps=10000"


def _add_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=_SET_HELP,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snake-drl", description="Modular deep RL on the batched snake engine."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train an agent")
    train.add_argument("-c", "--config", required=True, help="YAML config file")
    _add_overrides(train)

    evaluate = sub.add_parser("eval", help="evaluate a checkpoint")
    evaluate.add_argument("-m", "--model", required=True, help="checkpoint path")
    evaluate.add_argument("--episodes", type=int, default=None, help="episodes to average")
    evaluate.add_argument("--device", default=None, help="cpu | cuda | auto")
    _add_overrides(evaluate)

    play = sub.add_parser("play", help="play the game yourself (PyQt6)")
    play.add_argument("--width", type=int, default=16)
    play.add_argument("--height", type=int, default=16)
    play.add_argument("--seed", type=int, default=None)
    play.add_argument("--speed", type=int, default=4)
    play.add_argument("-m", "--model", default=None,
                      help="watch a trained checkpoint instead of playing yourself")
    play.add_argument("--record", default=None, metavar="GIF",
                      help="render the agent to a GIF and exit (no window, no PyQt6)")
    play.add_argument("--episodes", type=int, default=3,
                      help="episodes to record (with --record); default: 3")

    report = sub.add_parser("report", help="summarise a run and draw its curves")
    report.add_argument("run", help="run directory or metrics.jsonl")
    report.add_argument(
        "-o", "--out", default=None, help="output image (default: <run>/report.png)"
    )
    report.add_argument("--no-chart", action="store_true", help="print the summary only")

    bench = sub.add_parser("bench", help="measure env + policy throughput")
    bench.add_argument("-c", "--config", required=True, help="YAML config file")
    _add_overrides(bench)
    bench.add_argument("--steps", type=int, default=200, help="vector steps to time")
    bench.add_argument("--warmup", type=int, default=20)
    return parser


def cmd_train(args: argparse.Namespace) -> int:
    from .trainer import build_trainer

    cfg = load(args.config, args.overrides)
    trainer, run_dir = build_trainer(cfg)
    algorithm = trainer.algorithm
    policy = algorithm.policy
    print(
        f"run dir   : {run_dir}\n"
        f"device    : {algorithm.device}\n"
        f"algorithm : {type(cfg.algorithm).__name__} ({cfg.algorithm.type})\n"
        f"encoder   : {cfg.network.encoder.type} frames={cfg.network.encoder.frames}\n"
        f"policy    : {type(policy).__name__} "
        f"({sum(p.numel() for p in policy.parameters()) / 1e3:.1f}k params)\n"
        f"env       : {cfg.env.num_envs} x {cfg.env.width}x{cfg.env.height}\n"
        f"steps     : {cfg.train.total_steps}",
        flush=True,
    )
    try:
        trainer.run()
    finally:
        algorithm.close()
    print(f"done: {run_dir}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .trainer import build_evaluator, load_policy

    device = resolve_device(args.device or "auto")
    configure_runtime(device)
    policy, cfg, payload = load_policy(args.model, device, overrides=args.overrides)
    if args.episodes is not None:
        cfg.train.eval_episodes = args.episodes
    evaluator = build_evaluator(cfg, device)

    started = time.perf_counter()
    metrics = evaluator.run(policy)
    elapsed = time.perf_counter() - started
    print(f"checkpoint: {args.model} (trained {payload.get('env_steps', 0)} env steps)")
    for key, value in metrics.items():
        print(f"  {key:18s} {value:.4f}")
    print(f"  elapsed            {elapsed:.1f}s")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    # Recording renders offscreen and must not depend on PyQt6 being installed.
    if args.record is not None:
        if args.model is None:
            print("--record needs --model: there is nothing to record otherwise", file=sys.stderr)
            return 2
        from .play.record import record

        try:
            path = record(args.model, args.record, episodes=args.episodes)
        except (ValueError, FileNotFoundError) as exc:
            print(f"record failed: {exc}", file=sys.stderr)
            return 2
        print(f"recorded: {path}")
        return 0

    from .play.app import main as play_main

    argv = [
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--speed",
        str(args.speed),
    ]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.model is not None:
        argv += ["--model", args.model]
        from pathlib import Path

        if not Path(args.model).is_file():
            print(f"checkpoint not found: {args.model}", file=sys.stderr)
            return 2
    return play_main(argv)


def cmd_report(args: argparse.Namespace) -> int:
    from .report import (
        format_summary,
        load_metrics,
        plot_report,
        resolve_metrics_path,
        summarize,
    )

    metrics_path = resolve_metrics_path(args.run)
    rows = load_metrics(metrics_path)
    summary = summarize(rows)
    print(f"metrics : {metrics_path}")
    print(f"records : {len(rows)}")
    table = format_summary(summary)
    if table:
        print(table)
    else:
        print("no metrics to summarise")
    if args.no_chart or not rows:
        return 0
    out = args.out or str(metrics_path.parent / "report.png")
    written = plot_report(rows, out, title=str(metrics_path.parent.name))
    print(f"chart   : {written}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .envs import make_env
    from .nn.build import build_policy, count_parameters
    from .runner import VectorRunner

    cfg = load(args.config, args.overrides)
    device = resolve_device(cfg.run.device)
    runtime = configure_runtime(device, cfg.run.torch_threads)
    env = make_env(cfg.env, seed=cfg.env.seed)
    policy = build_policy(
        cfg.network,
        env.obs_spec,
        env.action_spec,
        family=cfg.policy_family,
        device=device,
        dueling=bool(getattr(cfg.algorithm, "is_dueling", False)),
    )
    runner = VectorRunner(env, policy.frames, device, pin_memory=cfg.run.pin_memory)

    def act(obs: torch.Tensor) -> torch.Tensor:
        if cfg.is_on_policy:
            return policy.act(obs).action
        return policy.act(obs, epsilon=0.05)

    runner.reset()
    with torch.inference_mode():
        for _ in range(args.warmup):
            runner.step(act(runner.obs))
        started = time.perf_counter()
        for _ in range(args.steps):
            runner.step(act(runner.obs))
        elapsed = time.perf_counter() - started

    total = args.steps * env.num_envs
    print(
        f"threads={runtime.torch_threads} device={device} encoder={cfg.network.encoder.type} "
        f"frames={policy.frames} params={count_parameters(policy) / 1e3:.1f}k\n"
        f"{total} env steps in {elapsed:.2f}s -> {total / elapsed:,.0f} env-step/s "
        f"({elapsed / args.steps * 1e3:.2f} ms per vectorized step, {env.num_envs} envs)"
    )
    runner.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {
        "train": cmd_train,
        "eval": cmd_eval,
        "play": cmd_play,
        "report": cmd_report,
        "bench": cmd_bench,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
