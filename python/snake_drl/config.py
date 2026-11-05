"""Typed configuration objects and YAML loading.

Design:

* every configurable component has one dataclass; unknown keys are rejected
  (``dacite`` strict mode) so a typo cannot silently fall back to a default;
* YAML loading/merging/CLI overrides use ``omegaconf`` instead of a
  hand-written parser;
* the algorithm block is dispatched on its ``type`` because PPO and the DQN
  family genuinely need different hyper-parameters.

All sizes default to something that runs on CPU; GPU users scale them up.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import dacite
from omegaconf import OmegaConf

__all__ = [
    "ALGORITHM_CONFIGS",
    "BufferConfig",
    "DQNConfig",
    "EncoderConfig",
    "EnvConfig",
    "NetworkConfig",
    "PPOConfig",
    "ProjectConfig",
    "RunConfig",
    "TrainConfig",
    "apply_overrides",
    "from_dict",
    "load",
    "to_dict",
]

# Values OmegaConf may hand back as strings; dacite coerces them.
_CASTS = [int, float, str]


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    """Environment selection and its generic knobs.

    ``kwargs`` carries environment-specific constructor arguments that do not
    deserve a top-level field (the toy environments use ``length`` and
    ``n_ctx``); the snake adapter ignores it.
    """

    type: str = "snake"
    num_envs: int = 256
    width: int = 16
    height: int = 16
    initial_length: int = 3
    max_episode_steps: int = 2000
    max_steps_without_food: int = 0
    food_reward: float = 1.0
    death_penalty: float = -1.0
    step_penalty: float = 0.0
    win_bonus: float = 2.0
    # "relative" -> 3 actions (straight/right/left); "absolute" -> 4 directions.
    action_mode: str = "relative"
    obs_mode: str = "grid"
    seed: int = 0
    num_threads: int = 0
    parallel_threshold: int = 128
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_mode not in ("relative", "absolute"):
            raise ValueError(
                f"env.action_mode must be 'relative' or 'absolute', got {self.action_mode!r}"
            )
        if self.obs_mode != "grid":
            raise ValueError(
                f"env.obs_mode={self.obs_mode!r} is not supported by the training adapter; "
                "only 'grid' is (bitboard->grid unpacking measured ~11x the copy it saves)"
            )
        if self.num_envs < 1:
            raise ValueError("env.num_envs must be >= 1")
        if not (3 <= self.width <= 64 and 3 <= self.height <= 64):
            raise ValueError("env.width/height must be within 3..64")


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
@dataclass
class EncoderConfig:
    """Flat config for all six encoders; only the fields of ``type`` are read.

    ==================  =====================================================
    type                reads
    ==================  =====================================================
    mlp                 hidden_dims, layer_norm, dropout
    cnn                 cnn_channels, cnn_kernel, cnn_pool
    gru                 gru_hidden, gru_layers, dropout
    transformer         tf_d_model, tf_layers, tf_heads, tf_ff, dropout
    convgru             cnn_channels, cnn_kernel, cnn_pool
    convtransformer     cnn_channels, cnn_kernel, cnn_pool, tf_d_model,
                        tf_layers, tf_heads, tf_ff, dropout
    ==================  =====================================================

    ``frames`` is the receptive window.  The temporal encoders
    (gru/transformer/convgru/convtransformer) require ``frames >= 2``; mlp/cnn
    accept ``frames > 1`` as plain frame stacking.
    """

    type: str = "cnn"
    frames: int = 1
    feature_dim: int = 256
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    layer_norm: bool = True
    dropout: float = 0.0
    cnn_channels: list[int] = field(default_factory=lambda: [32, 64, 64])
    cnn_kernel: int = 3
    cnn_pool: int = 2
    gru_hidden: int = 128
    gru_layers: int = 1
    tf_d_model: int = 128
    tf_layers: int = 2
    tf_heads: int = 4
    tf_ff: int = 256

    _TEMPORAL = ("gru", "transformer", "convgru", "convtransformer")
    _ALL = ("mlp", "cnn", "gru", "transformer", "convgru", "convtransformer")

    def __post_init__(self) -> None:
        if self.type not in self._ALL:
            raise ValueError(
                f"network.encoder.type must be one of {'/'.join(self._ALL)}, got {self.type!r}"
            )
        if self.frames < 1:
            raise ValueError("network.encoder.frames must be >= 1")
        if self.type in self._TEMPORAL and self.frames < 2:
            raise ValueError(f"encoder {self.type!r} is temporal; set network.encoder.frames >= 2")
        if self.feature_dim < 1:
            raise ValueError("network.encoder.feature_dim must be >= 1")
        if self.cnn_pool not in (1, 2, 4):
            raise ValueError("network.encoder.cnn_pool must be 1, 2 or 4")
        if not self.cnn_channels:
            raise ValueError("network.encoder.cnn_channels must not be empty")
        if self.tf_d_model % self.tf_heads:
            raise ValueError("network.encoder.tf_d_model must be divisible by tf_heads")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("network.encoder.dropout must be in [0, 1)")


@dataclass
class NetworkConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    # "auto" -> actor_critic for PPO, q/dueling_q for the DQN family.
    head: str = "auto"
    ortho_init: bool = True

    def __post_init__(self) -> None:
        if self.head not in ("auto", "actor_critic", "q", "dueling_q"):
            raise ValueError(
                f"network.head must be auto/actor_critic/q/dueling_q, got {self.head!r}"
            )


# --------------------------------------------------------------------------- #
# buffer
# --------------------------------------------------------------------------- #
@dataclass
class BufferConfig:
    type: str = "prioritized"  # "uniform" | "prioritized"
    capacity: int = 100_000
    alpha: float = 0.6
    beta0: float = 0.4
    beta_steps: int = 200_000
    eps: float = 1e-6
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    def __post_init__(self) -> None:
        if self.type not in ("uniform", "prioritized"):
            raise ValueError(f"buffer.type must be 'uniform' or 'prioritized', got {self.type!r}")
        if self.capacity < 1:
            raise ValueError("buffer.capacity must be >= 1")
        if self.alpha < 0.0:
            raise ValueError("buffer.alpha must be >= 0")
        if not (0.0 <= self.beta0 <= 1.0):
            raise ValueError("buffer.beta0 must be in [0, 1]")
        if self.beta_steps < 1:
            raise ValueError("buffer.beta_steps must be >= 1")


# --------------------------------------------------------------------------- #
# algorithms
# --------------------------------------------------------------------------- #
@dataclass
class PPOConfig:
    type: str = "ppo"
    lr: float = 3e-4
    n_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    clip_value_loss: bool = True
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    target_kl: float | None = None
    lr_schedule: str = "constant"  # "constant" | "linear"
    adam_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.n_steps < 1 or self.epochs < 0 or self.minibatch_size < 1:
            raise ValueError("ppo: n_steps and minibatch_size must be >= 1, epochs >= 0")
        if self.lr_schedule not in ("constant", "linear"):
            raise ValueError(
                f"ppo.lr_schedule must be 'constant' or 'linear', got {self.lr_schedule!r}"
            )


@dataclass
class DQNConfig:
    # dqn / ddqn / d3qn are presets of one algorithm; explicit double_q / dueling
    # values win over the preset.
    type: str = "d3qn"
    lr: float = 1e-4
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 1.0  # 1.0 = hard target sync, < 1 = soft update
    target_update: int = 1000  # env steps between target updates
    learning_starts: int = 5000  # env steps of random play before learning
    # One gradient step per ``train_freq`` collected transitions (the classic
    # "update every 4 env steps"); with vectorized environments this is not the
    # same as one update per vector step.
    train_freq: int = 4
    gradient_steps: int = 1  # gradient steps per trigger
    max_grad_norm: float = 10.0  # 0 disables clipping
    double_q: bool | None = None
    dueling: bool | None = None
    adam_eps: float = 1e-8
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000

    def __post_init__(self) -> None:
        if self.type not in ("dqn", "ddqn", "d3qn"):
            raise ValueError(f"algorithm.type must be one of dqn/ddqn/d3qn, got {self.type!r}")
        if self.batch_size < 1 or self.gradient_steps < 1 or self.train_freq < 1:
            raise ValueError("dqn: batch_size, gradient_steps and train_freq must be >= 1")
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError("dqn.tau must be in [0, 1]")
        if self.target_update < 1 or self.learning_starts < 0:
            raise ValueError("dqn: target_update must be >= 1 and learning_starts >= 0")

    @property
    def is_double(self) -> bool:
        if self.double_q is not None:
            return self.double_q
        return self.type in ("ddqn", "d3qn")

    @property
    def is_dueling(self) -> bool:
        if self.dueling is not None:
            return self.dueling
        return self.type == "d3qn"


ALGORITHM_CONFIGS: dict[str, type] = {
    "ppo": PPOConfig,
    "dqn": DQNConfig,
    "ddqn": DQNConfig,
    "d3qn": DQNConfig,
}


# --------------------------------------------------------------------------- #
# run / train
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    name: str = "run"
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:0"
    output_dir: str = "runs"
    # 0 = auto: a conservative value; torch's default is one thread per core and
    # oversubscribes next to the engine's own pool (measured up to 60x slower).
    torch_threads: int = 0
    pin_memory: bool = True


@dataclass
class TrainConfig:
    total_steps: int = 1_000_000
    log_every: int = 2_000
    eval_every: int = 50_000
    eval_episodes: int = 20
    eval_max_steps: int = 5_000
    save_every: int = 200_000
    keep_last: int = 3
    progress: bool = True

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise ValueError("train.total_steps must be >= 1")
        if min(self.log_every, self.eval_every, self.eval_episodes) < 1:
            raise ValueError("train.log_every, eval_every and eval_episodes must be >= 1")


@dataclass
class ProjectConfig:
    run: RunConfig = field(default_factory=RunConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    algorithm: PPOConfig | DQNConfig = field(default_factory=DQNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if not is_dataclass(self.algorithm):
            raise TypeError("algorithm must be a PPOConfig or DQNConfig")
        if isinstance(self.algorithm, PPOConfig) and self.algorithm.type != "ppo":
            raise ValueError("PPOConfig.type must be 'ppo'")
        if isinstance(self.algorithm, DQNConfig) and self.algorithm.type == "ppo":
            raise ValueError("DQNConfig.type must be dqn/ddqn/d3qn")

    @property
    def policy_family(self) -> str:
        """``"actor_critic"`` for PPO, ``"q"`` for the DQN family."""
        return "actor_critic" if isinstance(self.algorithm, PPOConfig) else "q"

    @property
    def is_on_policy(self) -> bool:
        return isinstance(self.algorithm, PPOConfig)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _algorithm_from_dict(raw: dict[str, Any]) -> PPOConfig | DQNConfig:
    kind = str(raw.get("type", "dqn")).lower()
    try:
        config_cls = ALGORITHM_CONFIGS[kind]
    except KeyError:
        available = ", ".join(sorted(ALGORITHM_CONFIGS))
        raise ValueError(f"unknown algorithm type {kind!r}; available: {available}") from None
    return dacite.from_dict(config_cls, raw, config=dacite.Config(strict=True, cast=_CASTS))


def from_dict(raw: dict[str, Any]) -> ProjectConfig:
    """Build a validated :class:`ProjectConfig` from a plain dictionary."""
    if not isinstance(raw, dict):
        raise TypeError(f"config root must be a mapping, got {type(raw).__name__}")
    data = dict(raw)
    algorithm = _algorithm_from_dict(data.pop("algorithm", {}))
    cfg = dacite.from_dict(ProjectConfig, data, config=dacite.Config(strict=True, cast=_CASTS))
    cfg.algorithm = algorithm
    return cfg


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key.path=value`` overrides to a raw config dictionary."""
    if not overrides:
        return raw
    merged = OmegaConf.merge(OmegaConf.create(raw), OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


def load(path: str | None = None, overrides: list[str] | None = None) -> ProjectConfig:
    """Load YAML (optional) and apply ``key.path=value`` overrides.

    Example::

        cfg = load("configs/dqn_cnn.yaml", ["train.total_steps=10000"])
    """
    raw: dict[str, Any] = {}
    if path:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[assignment]
    return from_dict(apply_overrides(dict(raw), list(overrides or [])))


def to_dict(obj: Any) -> dict[str, Any]:
    """Plain, JSON/YAML-safe view of a dataclass config."""
    if not is_dataclass(obj):
        raise TypeError(f"expected a dataclass config, got {type(obj).__name__}")
    out: dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if is_dataclass(value):
            out[f.name] = to_dict(value)
        elif isinstance(value, tuple):
            out[f.name] = list(value)
        else:
            out[f.name] = value
    return out
