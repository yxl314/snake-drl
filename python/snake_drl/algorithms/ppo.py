"""Proximal policy optimization.

Rollouts continue across iterations: every environment keeps its episode until
the environment itself ends it.  Resetting at the start of each rollout would cut
episodes at ``n_steps`` and throw away the agents' progress.
"""

from __future__ import annotations

from typing import Any

import torch

from ..buffers.rollout import RolloutBuffer
from ..config import PPOConfig
from ..nn.policies import ActorCriticPolicy
from ..runner import VectorRunner
from .base import Algorithm
from .gae import compute_gae
from .optim import optimizer_step

__all__ = ["PPO"]

# Minibatch statistics, in a fixed order (one stack + one host sync per update).
_STAT_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")


class PPO(Algorithm):
    policy: ActorCriticPolicy

    def __init__(
        self,
        cfg: PPOConfig,
        policy: ActorCriticPolicy,
        runner: VectorRunner,
        buffer: RolloutBuffer,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        total_steps: int | None = None,
    ) -> None:
        super().__init__(policy, runner, device, total_steps=total_steps)
        self.cfg = cfg
        self.buffer = buffer
        self.optimizer = optimizer
        self._minibatch = min(cfg.minibatch_size, cfg.n_steps * runner.num_envs)
        self._mb_obs = torch.empty(
            (self._minibatch, *buffer.obs.shape[2:]),
            dtype=buffer.obs.dtype,
            device=device,
        )
        self._obs: torch.Tensor | None = None

    # ------------------------------------------------------------------ core
    def _iterate(self) -> dict[str, float]:
        num_envs = self.runner.num_envs
        self.buffer.reset()
        self.policy.eval()
        with torch.no_grad():
            obs = self._obs if self._obs is not None else self.runner.reset()
            for _ in range(self.cfg.n_steps):
                out = self.policy.act(obs)
                step = self.runner.step(out.action)
                self.buffer.add(
                    obs,
                    step.next_obs,
                    out.action,
                    step.reward,
                    out.logprob,
                    out.value,
                    step.terminated,
                    step.truncated,
                )
                obs = self.runner.obs
            last_next = step.next_obs
        self._obs = obs
        self.env_steps += num_envs * self.cfg.n_steps
        self.policy.train()
        return self._update(last_next)

    # ---------------------------------------------------------------- update
    def _update(self, last_next: torch.Tensor) -> dict[str, float]:
        advantages, returns = self._advantages(last_next)
        adv = advantages.reshape(-1)
        ret = returns.reshape(-1)
        if self.cfg.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        lr = self._current_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        rows = self._run_epochs(self.buffer.flat(), adv, ret)
        values = torch.stack(rows).mean(dim=0).tolist()
        metrics = dict(zip(_STAT_KEYS, values, strict=True))
        metrics["updates"] = float(len(rows))
        metrics["lr"] = lr
        return metrics

    def _advantages(self, last_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE with the correct successor value on every transition.

        ``V(s')`` is the next stored value for continuing transitions, the final
        observation's value for truncated ones (only that subset is re-evaluated)
        and unused for terminations.
        """
        buf = self.buffer
        with torch.no_grad():
            next_values = torch.empty_like(buf.values)
            next_values[:-1].copy_(buf.values[1:])
            next_values[-1].copy_(self.policy.value(last_next))
            if buf.truncated.any():
                rows = buf.truncated.nonzero(as_tuple=True)
                next_values[rows] = self.policy.value(buf.next_obs[rows])
            return compute_gae(
                buf.rewards,
                buf.values,
                next_values,
                buf.terminated,
                self.cfg.gamma,
                self.cfg.gae_lambda,
            )

    def _run_epochs(
        self, flat: dict[str, torch.Tensor], adv: torch.Tensor, ret: torch.Tensor
    ) -> list[torch.Tensor]:
        """Minibatch epochs; returns one statistics row per optimizer step."""
        cfg = self.cfg
        total = adv.shape[0]
        minibatch = self._minibatch
        parameters = self.policy.parameters()
        rows: list[torch.Tensor] = []
        permutation = torch.randperm(total, device=self.device)
        stop = False
        for _ in range(cfg.epochs):
            if stop:
                break
            for start in range(0, total, minibatch):
                idx = permutation[start : start + minibatch]
                n = idx.numel()
                mb_obs = torch.index_select(flat["obs"], 0, idx, out=self._mb_obs[:n])
                logprob, entropy, value = self.policy.evaluate_actions(mb_obs, flat["action"][idx])
                old_logprob = flat["logprob"][idx]
                old_value = flat["value"][idx]
                mb_adv = adv[idx]
                mb_ret = ret[idx]

                log_ratio = logprob - old_logprob
                ratio = log_ratio.exp()
                surrogate = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio),
                ).mean()
                value_loss = self._value_loss(value, old_value, mb_ret)
                loss = surrogate + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

                optimizer_step(self.optimizer, parameters, loss, max_grad_norm=cfg.max_grad_norm)
                self.updates += 1

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    rows.append(
                        torch.stack(
                            (
                                surrogate.detach(),
                                value_loss.detach(),
                                entropy.mean().detach(),
                                approx_kl.detach(),
                                ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean(),
                            )
                        )
                    )
                if cfg.target_kl is not None and float(approx_kl) > 1.5 * cfg.target_kl:
                    stop = True
                    break
        return rows

    def _value_loss(
        self, value: torch.Tensor, old_value: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.cfg
        if cfg.clip_value_loss:
            clipped = old_value + (value - old_value).clamp(-cfg.clip_ratio, cfg.clip_ratio)
            return 0.5 * torch.max((value - target).pow(2), (clipped - target).pow(2)).mean()
        return 0.5 * (value - target).pow(2).mean()

    def _current_lr(self) -> float:
        cfg = self.cfg
        if cfg.lr_schedule == "constant" or not self.total_steps:
            return cfg.lr
        progress = min(1.0, self.env_steps / max(1, self.total_steps))
        return cfg.lr * (1.0 - progress)

    # ------------------------------------------------------------ lifecycle
    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
