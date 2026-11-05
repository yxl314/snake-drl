"""Deterministic evaluation of a policy.

Runs on its own environment instance so evaluation never disturbs the training
stream, and tracks episode returns/lengths itself so it works for any
environment -- including ones that do not expose scores.
"""

from __future__ import annotations

import statistics
from typing import Callable, Dict, List

import torch

from ..envs import VectorEnv
from ..runner import VectorRunner


class Evaluator:
    def __init__(
        self,
        make_env: Callable[[], VectorEnv],
        device: torch.device | str,
        *,
        episodes: int = 20,
        max_steps: int = 20_000,
    ) -> None:
        self.make_env = make_env
        self.device = torch.device(device)
        self.episodes = int(episodes)
        self.max_steps = int(max_steps)

    @torch.no_grad()
    def run(self, policy: torch.nn.Module) -> Dict[str, float]:
        env = self.make_env()
        runner = VectorRunner(env, policy.frames, self.device, pin_memory=False)
        was_training = policy.training
        policy.eval()
        try:
            runner.reset()
            returns = torch.zeros(env.num_envs, dtype=torch.float32, device=self.device)
            lengths = torch.zeros(env.num_envs, dtype=torch.int64, device=self.device)
            finished_returns: List[float] = []
            finished_lengths: List[int] = []
            steps = 0
            while len(finished_returns) < self.episodes and steps < self.max_steps:
                action = policy.greedy(runner.obs)
                step = runner.step(action)
                steps += env.num_envs
                returns += step.reward
                lengths += 1
                done = step.terminated | step.truncated
                if done.any():
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
