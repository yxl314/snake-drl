"""Proximal policy optimization."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..buffers.rollout import RolloutBuffer
from ..config import PPOConfig
from ..runner import VectorRunner
from .base import Algorithm
from .gae import compute_gae


class PPO(Algorithm):
    def __init__(
        self,
        cfg: PPOConfig,
        policy: nn.Module,
        runner: VectorRunner,
        buffer: RolloutBuffer,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        super().__init__(policy, runner, device)
        self.cfg = cfg
        self.buffer = buffer
        self.optimizer = optimizer
        self._minibatch = min(cfg.minibatch_size, cfg.n_steps * runner.num_envs)
        self._mb_obs = torch.empty(
            (self._minibatch, *buffer.obs.shape[2:]), dtype=buffer.obs.dtype, device=device
        )

    # ------------------------------------------------------------------ core
    def iterate(self) -> Dict[str, float]:
        num_envs = self.runner.num_envs
        self.buffer.reset()
        self.policy.eval()
        with torch.no_grad():
            obs = self.runner.reset()
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
        self.env_steps += num_envs * self.cfg.n_steps
        self.policy.train()
        return self._update(last_next)

    # ---------------------------------------------------------------- update
    def _update(self, last_next: torch.Tensor) -> Dict[str, float]:
        cfg = self.cfg
        buf = self.buffer
        with torch.no_grad():
            next_values = torch.empty_like(buf.values)
            next_values[:-1].copy_(buf.values[1:])
            next_values[-1].copy_(self.policy.value(last_next))
            if buf.truncated.any():
                rows = buf.truncated.nonzero(as_tuple=True)
                next_values[rows] = self.policy.value(buf.next_obs[rows])
            advantages, returns = compute_gae(
                buf.rewards, buf.values, next_values, buf.terminated, cfg.gamma, cfg.gae_lambda
            )

        flat = buf.flat()
        adv = advantages.reshape(-1)
        ret = returns.reshape(-1)
        if cfg.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        total = adv.shape[0]
        minibatch = self._minibatch
        lr = self._current_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")}
        perm = torch.randperm(total, device=self.device)
        stop = False
        for _ in range(cfg.epochs):
            if stop:
                break
            for start in range(0, total, minibatch):
                idx = perm[start : start + minibatch]
                n = idx.numel()
                mb_obs = torch.index_select(flat["obs"], 0, idx, out=self._mb_obs[:n])
                logprob, entropy, value = self.policy.evaluate_actions(
                    mb_obs, flat["action"][idx]
                )
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

                if cfg.clip_value_loss:
                    value_clipped = old_value + (value - old_value).clamp(
                        -cfg.clip_ratio, cfg.clip_ratio
                    )
                    value_loss = 0.5 * torch.max(
                        (value - mb_ret).pow(2), (value_clipped - mb_ret).pow(2)
                    ).mean()
                else:
                    value_loss = 0.5 * (value - mb_ret).pow(2).mean()

                loss = surrogate + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    stats["policy_loss"].append(surrogate.detach())
                    stats["value_loss"].append(value_loss.detach())
                    stats["entropy"].append(entropy.mean().detach())
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().detach()
                    stats["approx_kl"].append(approx_kl)
                    stats["clip_frac"].append(
                        ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean().detach()
                    )
                if cfg.target_kl is not None and float(approx_kl) > 1.5 * cfg.target_kl:
                    stop = True
                    break
        self.updates += 1
        return {
            "policy_loss": float(torch.stack(stats["policy_loss"]).mean()),
            "value_loss": float(torch.stack(stats["value_loss"]).mean()),
            "entropy": float(torch.stack(stats["entropy"]).mean()),
            "approx_kl": float(torch.stack(stats["approx_kl"]).mean()),
            "clip_frac": float(torch.stack(stats["clip_frac"]).mean()),
            "lr": lr,
        }

    def _current_lr(self) -> float:
        cfg = self.cfg
        if cfg.lr_schedule == "constant" or not self.total_steps:
            return cfg.lr
        progress = min(1.0, self.env_steps / max(1, self.total_steps))
        return cfg.lr * (1.0 - progress)

    # ------------------------------------------------------------ lifecycle
    def state_dict(self) -> Dict[str, object]:
        state = super().state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, object]) -> None:
        super().load_state_dict(state)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
