"""DQN / Double DQN / Dueling Double DQN in one implementation.

The three names are presets of the same update:

=================  ==============  ===============
preset             double_q        dueling
=================  ==============  ===============
``dqn``            no              no
``ddqn``           yes             no
``d3qn``           yes             yes
=================  ==============  ===============

``double_q`` decouples action selection (online net) from evaluation (target
net), which is exactly what DDQN adds.  ``dueling`` is a head choice and is
therefore applied when the policy is built; this class only sees a Q policy.

Termination vs truncation: the target bootstraps whenever ``terminated`` is
false, including truncated steps, using the final observation stored as
``next_obs``.  Treating a time limit as a death would bias Q downwards.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from ..buffers.base import ReplayProtocol
from ..config import DQNConfig
from ..runner import VectorRunner
from .base import Algorithm
from .exploration import BetaFn, EpsilonFn


def q_target(
    q_next_target: torch.Tensor,
    q_next_online: torch.Tensor,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    double_q: bool,
) -> torch.Tensor:
    """Bootstrap target: ``r + gamma * (1 - terminated) * bootstrap``.

    ``double_q`` picks the next action with the online net but evaluates it with
    the target net (DDQN).  Non-double uses the target net for both.
    """
    if double_q:
        best = q_next_online.argmax(dim=1, keepdim=True)
        bootstrap = q_next_target.gather(1, best).squeeze(1)
    else:
        bootstrap = q_next_target.max(dim=1).values
    return reward + gamma * (~terminated).to(torch.float32) * bootstrap


class DQN(Algorithm):
    def __init__(
        self,
        cfg: DQNConfig,
        policy: torch.nn.Module,
        target_policy: torch.nn.Module,
        runner: VectorRunner,
        replay: ReplayProtocol,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        epsilon: EpsilonFn,
        beta: BetaFn,
    ) -> None:
        super().__init__(policy, runner, device)
        self.cfg = cfg
        self.target_policy = target_policy
        self.replay = replay
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.beta = beta
        self._obs: torch.Tensor | None = None
        self._next_target = cfg.target_update
        # credits are counted in transitions so the update rate does not depend
        # on how many environments are stepped at once
        self._credit = 0
        self.target_policy.eval()
        for param in self.target_policy.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------ core
    def iterate(self) -> Dict[str, float]:
        self.policy.eval()
        obs = self._obs if self._obs is not None else self.runner.reset()
        num_envs = self.runner.num_envs
        with torch.no_grad():
            epsilon = self.epsilon(self.env_steps)
            action = self.policy.act(obs, epsilon)
            step = self.runner.step(action)
            self.replay.add(
                obs,
                step.next_obs,
                action,
                step.reward,
                step.terminated,
                step.truncated,
            )
            obs = self.runner.obs
        self._obs = obs
        self.env_steps += num_envs

        metrics: Dict[str, float] = {"epsilon": epsilon}
        ready = len(self.replay) >= max(self.cfg.learning_starts, self.cfg.batch_size)
        self._credit += num_envs
        updates = 0
        if ready:
            updates = int(self._credit // self.cfg.train_freq) * self.cfg.gradient_steps
            self._credit -= (updates // self.cfg.gradient_steps) * self.cfg.train_freq
        else:
            # do not bank a burst of updates for after learning_starts
            self._credit = min(self._credit, self.cfg.train_freq * 4)
        if updates:
            self.policy.train()
            # one device->host sync per iteration instead of one per statistic
            stats = torch.stack([self._update() for _ in range(updates)])
            mean = stats.mean(dim=0).tolist()
            metrics.update(
                {"loss": mean[0], "q_mean": mean[1], "target_mean": mean[2], "td_abs_mean": mean[3]}
            )
        metrics["updates"] = float(updates)
        self._maybe_sync_target()
        return metrics

    # ---------------------------------------------------------------- update
    def _update(self) -> torch.Tensor:
        """One gradient step; returns ``[loss, q_mean, target_mean, |td|]`` (device)."""
        cfg = self.cfg
        batch = self.replay.sample(cfg.batch_size, self.beta(self.env_steps))
        q = self.policy(batch.obs).gather(1, batch.action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            target = q_target(
                self.target_policy(batch.next_obs),
                self.policy(batch.next_obs),
                batch.reward,
                batch.terminated,
                cfg.gamma,
                cfg.is_double,
            )

        td_error = q - target
        loss = (batch.weights * F.smooth_l1_loss(q, target, reduction="none")).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip and cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        self.replay.update_priorities(batch.indices, td_error.detach())
        self.updates += 1
        return torch.stack(
            (loss.detach(), q.mean(), target.mean(), td_error.abs().mean())
        )

    def _maybe_sync_target(self) -> None:
        cfg = self.cfg
        if cfg.tau < 1.0:
            with torch.no_grad():
                for target, online in zip(self.target_policy.parameters(), self.policy.parameters()):
                    target.mul_(1.0 - cfg.tau).add_(online, alpha=cfg.tau)
            return
        if self.env_steps >= self._next_target:
            self.target_policy.load_state_dict(self.policy.state_dict())
            self._next_target = self.env_steps + cfg.target_update

    # ------------------------------------------------------------ lifecycle
    def state_dict(self) -> Dict[str, object]:
        state = super().state_dict()
        state["target_policy"] = self.target_policy.state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        state["next_target"] = self._next_target
        return state

    def load_state_dict(self, state: Dict[str, object]) -> None:
        super().load_state_dict(state)
        self.target_policy.load_state_dict(state["target_policy"])  # type: ignore[arg-type]
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
        self._next_target = int(state.get("next_target", self.cfg.target_update))
