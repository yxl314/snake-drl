"""Deterministic evaluation of a policy.

Evaluation runs on its own environment batch and its own seed, so it never
disturbs the training stream.  Episode returns and lengths are tracked here
rather than in the environment, which makes the evaluator work with any
environment -- including ones that do not expose a score.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

import torch

from ..envs import VectorEnv
from ..runner import VectorRunner

__all__ = ["Evaluator"]


class Evaluator:
    def __init__(
        self,
        make_env: Callable[[], VectorEnv],
        device: torch.device | str,
        *,
        episodes: int = 20,
        max_steps: int = 20_000,
    ) -> None:
        if episodes < 1:
            raise ValueError(f"episodes must be >= 1, got {episodes}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        self.make_env = make_env
        self.device = torch.device(device)
        self.episodes = int(episodes)
        self.max_steps = int(max_steps)

    @torch.no_grad()
    def run(self, policy: torch.nn.Module) -> dict[str, float]:
        """Run greedy episodes and return ``eval/*`` metrics.

        The policy must expose ``frames`` and ``greedy(obs)`` (both built-in
        policies do).
        """
        env = self.make_env()
        runner = VectorRunner(env, policy.frames, self.device, pin_memory=False)
        was_training = policy.training
        policy.eval()
        try:
            runner.reset()
            returns = torch.zeros(env.num_envs, dtype=torch.float32, device=self.device)
            lengths = torch.zeros(env.num_envs, dtype=torch.int64, device=self.device)
            finished_returns: list[float] = []
            finished_lengths: list[int] = []
            steps = 0
            while len(finished_returns) < self.episodes and steps < self.max_steps:
                action = policy.greedy(runner.obs)
                step = runner.step(action)
                steps += env.num_envs
                returns += step.reward
                lengths += 1
                done = step.terminated | step.truncated
                if done.any():
                    # The batch can overshoot the requested episode count; that
                    # only makes the estimate slightly better.
                    finished_returns.extend(returns[done].tolist())
                    finished_lengths.extend(lengths[done].tolist())
                    returns[done] = 0.0
                    lengths[done] = 0
            if not finished_returns:
                return {"eval/episodes": 0.0, "eval/return": float("nan")}
            return {
                "eval/episodes": float(len(finished_returns)),
                "eval/return": statistics.fmean(finished_returns),
                "eval/return_std": (
                    statistics.pstdev(finished_returns) if len(finished_returns) > 1 else 0.0
                ),
                "eval/return_max": max(finished_returns),
                "eval/length": statistics.fmean(finished_lengths),
            }
        finally:
            runner.close()
            policy.train(was_training)
