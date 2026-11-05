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

import torch
import torch.nn.functional as F

from ..buffers.protocols import ReplayProtocol
from ..config import DQNConfig
from ..runner import VectorRunner
from .base import Algorithm
from .exploration import BetaFn, EpsilonFn
from .optim import optimizer_step

__all__ = ["DQN", "q_target"]

# Statistics returned by one gradient step, in a fixed order.
_STAT_KEYS = ("loss", "q_mean", "target_mean", "td_abs_mean")

# How many update triggers may be banked while the buffer is still filling up;
# without the cap the first learnable iteration would fire a burst.
_MAX_BANKED_TRIGGERS = 4


def q_target(
    q_next_target: torch.Tensor,
    q_next_online: torch.Tensor | None,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    double_q: bool,
) -> torch.Tensor:
    """Bootstrap target: ``r + gamma * (1 - terminated) * bootstrap``.

    ``double_q`` picks the next action with the online net but evaluates it with
    the target net; ``q_next_online`` may be ``None`` when ``double_q`` is false,
    which saves a full forward pass in plain DQN.
    """
    if double_q:
        if q_next_online is None:
            raise ValueError("double_q=True requires the online next-state Q values")
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
        total_steps: int | None = None,
    ) -> None:
        super().__init__(policy, runner, device, total_steps=total_steps)
        self.cfg = cfg
        self.target_policy = target_policy
        self.replay = replay
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.beta = beta
        self._obs: torch.Tensor | None = None
        self._next_target = cfg.target_update
        # Update credit is counted in transitions, so the update rate does not
        # silently shrink when ``num_envs`` grows.
        self._credit = 0
        self.target_policy.eval()
        for param in self.target_policy.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------ core
    def _iterate(self) -> dict[str, float]:
        self.policy.eval()
        num_envs = self.runner.num_envs
        obs = self._obs if self._obs is not None else self.runner.reset()
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
        self._obs = self.runner.obs
        self.env_steps += num_envs

        metrics: dict[str, float] = {"epsilon": epsilon}
        updates = self._credit_updates(num_envs)
        if updates:
            self.policy.train()
            # One device->host synchronization per iteration instead of one per
            # statistic.
            stats = torch.stack([self._update() for _ in range(updates)]).mean(dim=0)
            values = stats.tolist()
            metrics.update(dict(zip(_STAT_KEYS, values, strict=True)))
        metrics["updates"] = float(updates)
        self._maybe_sync_target()
        return metrics

    def _credit_updates(self, num_envs: int) -> int:
        """Number of gradient steps earned by ``num_envs`` new transitions."""
        cfg = self.cfg
        ready = len(self.replay) >= max(cfg.learning_starts, cfg.batch_size)
        self._credit += num_envs
        if not ready:
            self._credit = min(self._credit, cfg.train_freq * _MAX_BANKED_TRIGGERS)
            return 0
        triggers = self._credit // cfg.train_freq
        self._credit -= triggers * cfg.train_freq
        return triggers * cfg.gradient_steps

    # ---------------------------------------------------------------- update
    def _update(self) -> torch.Tensor:
        """One gradient step; returns ``[loss, q_mean, target_mean, |td|]``."""
        cfg = self.cfg
        batch = self.replay.sample(cfg.batch_size, self.beta(self.env_steps))
        batch = batch.to(self.device, non_blocking=True)
        q = self.policy(batch.obs).gather(1, batch.action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next_online = self.policy(batch.next_obs) if cfg.is_double else None
            target = q_target(
                self.target_policy(batch.next_obs),
                q_next_online,
                batch.reward,
                batch.terminated,
                cfg.gamma,
                cfg.is_double,
            )

        td_error = q - target
        loss = (batch.weights * F.smooth_l1_loss(q, target, reduction="none")).mean()
        optimizer_step(
            self.optimizer,
            self.policy.parameters(),
            loss,
            max_grad_norm=cfg.max_grad_norm,
        )
        self.replay.update_priorities(batch.indices, td_error.detach())
        self.updates += 1
        return torch.stack((loss.detach(), q.mean(), target.mean(), td_error.abs().mean()))

    def _maybe_sync_target(self) -> None:
        cfg = self.cfg
        if cfg.tau < 1.0:
            with torch.no_grad():
                for target, online in zip(
                    self.target_policy.parameters(), self.policy.parameters(), strict=True
                ):
                    target.mul_(1.0 - cfg.tau).add_(online, alpha=cfg.tau)
            return
        if self.env_steps >= self._next_target:
            self.target_policy.load_state_dict(self.policy.state_dict())
            self._next_target = self.env_steps + cfg.target_update

    # ------------------------------------------------------------ lifecycle
    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["target_policy"] = self.target_policy.state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        state["next_target"] = self._next_target
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        self.target_policy.load_state_dict(state["target_policy"])  # type: ignore[arg-type]
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
        self._next_target = int(state.get("next_target", self.cfg.target_update))
