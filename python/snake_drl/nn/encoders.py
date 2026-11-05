"""Six interchangeable sequence encoders.

Every encoder consumes ``(B, T, *frame_shape)`` and returns ``(B, feature_dim)``.
Keeping the time axis inside the encoder is what lets PPO and the DQN family
share one policy/buffer contract: the algorithm always sees "one observation per
environment per step", and the encoder decides how much history it carries.

``T`` belongs to the encoder (``frames``), not to the algorithm: the runner
maintains a rolling window of exactly that many frames.

Measured cost (B=256, T=4, 3x16x16, 4 torch threads, aarch64 CPU):

=================  ==========  ===========  ============
encoder            fwd (ms)    train (ms)   params
=================  ==========  ===========  ============
mlp                     8.1         15.0        886k
cnn                     8.1         25.8         92k
gru                    13.7         20.3        214k
transformer            31.9         61.6        381k
convgru                91.0        316.3        250k
convtransformer       416.8        831.7        323k
=================  ==========  ===========  ============

That gap is why the CPU presets use cnn/mlp; the convolutional recurrent and
transformer variants are for a GPU or a small board.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

from ..config import EncoderConfig
from ..registry import Registry
from ..spaces import ObsSpec
from .layers import FastLinear

ENCODERS: Registry[type] = Registry("encoder")


def _group_norm(channels: int) -> nn.Module:
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


def _conv_out(size: int, layers: int, pool: int) -> int:
    for _ in range(layers):
        size = math.ceil(size / pool)
    return max(1, size)


class Encoder(nn.Module):
    """Common interface: ``frames`` describes the window the runner must keep."""

    frames: int
    feature_dim: int
    accepts: tuple[str, ...] = ("image", "vector")

    def __init__(self, frames: int, feature_dim: int) -> None:
        super().__init__()
        self.frames = int(frames)
        self.feature_dim = int(feature_dim)

    @staticmethod
    def check_spec(spec: ObsSpec, accepts: tuple[str, ...], name: str) -> None:
        if spec.kind not in accepts:
            raise ValueError(
                f"encoder {name!r} accepts {accepts} observations, got {spec.kind!r}"
            )


@ENCODERS.register("mlp")
class MLPEncoder(Encoder):
    accepts = ("image", "vector")

    def __init__(self, spec: ObsSpec, cfg: EncoderConfig) -> None:
        super().__init__(cfg.frames, cfg.feature_dim)
        self.check_spec(spec, self.accepts, "mlp")
        width = spec.frame_elems * cfg.frames
        layers: List[nn.Module] = []
        prev = width
        for hidden in cfg.hidden_dims:
            layers.append(FastLinear(prev, hidden))
            if cfg.layer_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev = hidden
        layers.append(FastLinear(prev, cfg.feature_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.flatten(1))


@ENCODERS.register("cnn")
class CNNEncoder(Encoder):
    accepts = ("image",)

    def __init__(self, spec: ObsSpec, cfg: EncoderConfig) -> None:
        super().__init__(cfg.frames, cfg.feature_dim)
        self.check_spec(spec, self.accepts, "cnn")
        channels, height, width = spec.shape
        blocks: List[nn.Module] = []
        prev = channels * cfg.frames  # frame stacking lives on the channel axis
        for ch in cfg.cnn_channels:
            blocks += [
                nn.Conv2d(prev, ch, cfg.cnn_kernel, stride=cfg.cnn_pool, padding=1),
                _group_norm(ch),
                nn.SiLU(),
            ]
            prev = ch
        self.body = nn.Sequential(*blocks)
        out_h = min(2, _conv_out(height, len(cfg.cnn_channels), cfg.cnn_pool))
        out_w = min(2, _conv_out(width, len(cfg.cnn_channels), cfg.cnn_pool))
        self.pool = nn.AdaptiveAvgPool2d((out_h, out_w))
        self.head = nn.Sequential(
            nn.Flatten(),
            FastLinear(prev * out_h * out_w, cfg.feature_dim),
            nn.LayerNorm(cfg.feature_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch, frames = obs.shape[:2]
        x = obs.reshape(batch, frames * obs.shape[2], *obs.shape[3:])
        return self.head(self.pool(self.body(x)))


@ENCODERS.register("gru")
class GRUEncoder(Encoder):
    accepts = ("image", "vector")

    def __init__(self, spec: ObsSpec, cfg: EncoderConfig) -> None:
        super().__init__(cfg.frames, cfg.feature_dim)
        self.check_spec(spec, self.accepts, "gru")
        self.inp = FastLinear(spec.frame_elems, cfg.gru_hidden)
        self.gru = nn.GRU(
            cfg.gru_hidden,
            cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.gru_hidden), FastLinear(cfg.gru_hidden, cfg.feature_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch, frames = obs.shape[:2]
        _, hidden = self.gru(self.inp(obs.reshape(batch, frames, -1)))
        return self.head(hidden[-1])


@ENCODERS.register("transformer")
class TransformerEncoder(Encoder):
    accepts = ("image", "vector")

    def __init__(self, spec: ObsSpec, cfg: EncoderConfig) -> None:
        super().__init__(cfg.frames, cfg.feature_dim)
        self.check_spec(spec, self.accepts, "transformer")
        self.inp = FastLinear(spec.frame_elems, cfg.tf_d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.frames, cfg.tf_d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.tf_d_model,
            nhead=cfg.tf_heads,
            dim_feedforward=cfg.tf_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False: the nested path is slower for short windows.
        self.body = nn.TransformerEncoder(layer, cfg.tf_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.tf_d_model), FastLinear(cfg.tf_d_model, cfg.feature_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch, frames = obs.shape[:2]
        z = self.inp(obs.reshape(batch, frames, -1)) + self.pos[:, :frames]
        return self.head(self.body(z)[:, -1])


@ENCODERS.register("convgru")
class ConvGRUEncoder(Encoder):
    """Convolutional GRU: a 2-D hidden state updated over the frame window."""

    accepts = ("image",)

    def __init__(self, spec: ObsSpec, cfg: EncoderConfig) -> None:
        super().__init__(cfg.frames, cfg.feature_dim)
        self.check_spec(spec, self.accepts, "convgru")
        channels = spec.shape[0]
        stem: List[nn.Module] = []
        prev = channels
        for ch in cfg.cnn_channels:
            stem += [
                nn.Conv2d(prev, ch, cfg.cnn_kernel, stride=cfg.cnn_pool, padding=1),
                _group_norm(ch),
                nn.SiLU(),
            ]
            prev = ch
        self.stem = nn.Sequential(*stem)
        self.hidden = cfg.cnn_channels[-1]
        self.gate_x = nn.Conv2d(self.hidden, self.hidden * 3, 3, padding=1)
        self.gate_h = nn.Conv2d(self.hidden, self.hidden * 3, 3, padding=1)
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden), FastLinear(self.hidden, cfg.feature_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch, frames = obs.shape[:2]
        feats = self.stem(obs.reshape(batch * frames, *obs.shape[2:]))
        feats = feats.reshape(batch, frames, *feats.shape[1:])
        state = torch.zeros_like(feats[:, 0])
        h = self.hidden
        for t in range(frames):
            gx, gh = self.gate_x(feats[:, t]), self.gate_h(state)
            r = torch.sigmoid(gx[:, :h] + gh[:, :h])
            z = torch.sigmoid(gx[:, h : 2 * h] + gh[:, h : 2 * h])
            n = torch.tanh(gx[:, 2 * h :] + r * gh[:, 2 * h :])
            state = (1.0 - z) * n + z * state
        return self.head(state.mean(dim=(2, 3)))


@ENCODERS.register("convtransformer")
class ConvTransformerEncoder(Encoder):
    """Conv patch embedding + a transformer over (time x patch) tokens."""

    accepts = ("image",)

    def __init__(self, spec: ObsSpec, cfg: EncoderConfig) -> None:
        super().__init__(cfg.frames, cfg.feature_dim)
        self.check_spec(spec, self.accepts, "convtransformer")
        channels, height, width = spec.shape
        stem: List[nn.Module] = []
        prev = channels
        for ch in cfg.cnn_channels:
            stem += [
                nn.Conv2d(prev, ch, cfg.cnn_kernel, stride=cfg.cnn_pool, padding=1),
                _group_norm(ch),
                nn.SiLU(),
            ]
            prev = ch
        self.stem = nn.Sequential(*stem)
        self.n_tokens = _conv_out(height, len(cfg.cnn_channels), cfg.cnn_pool) * _conv_out(
            width, len(cfg.cnn_channels), cfg.cnn_pool
        )
        self.proj = nn.Conv2d(prev, cfg.tf_d_model, 1)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, cfg.tf_d_model))
        self.frame_emb = nn.Parameter(torch.zeros(1, cfg.frames, 1, cfg.tf_d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.tf_d_model,
            nhead=cfg.tf_heads,
            dim_feedforward=cfg.tf_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, cfg.tf_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.tf_d_model), FastLinear(cfg.tf_d_model, cfg.feature_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch, frames = obs.shape[:2]
        feats = self.proj(self.stem(obs.reshape(batch * frames, *obs.shape[2:])))
        tokens = feats.flatten(2).transpose(1, 2).reshape(batch, frames, self.n_tokens, -1)
        z = tokens + self.pos[:, None] + self.frame_emb[:, :frames]
        z = self.body(z.reshape(batch, frames * self.n_tokens, -1))
        return self.head(z.mean(dim=1))


def build_encoder(spec: ObsSpec, cfg: EncoderConfig) -> Encoder:
    """Instantiate the encoder named by ``cfg.type``."""
    factory = ENCODERS.get(cfg.type)
    return factory(spec, cfg)  # type: ignore[return-value]
