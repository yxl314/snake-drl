"""Conv patch embedding followed by attention over (time x patch) tokens."""

from __future__ import annotations

import torch
import torch.nn as nn

from ...config import EncoderConfig
from ...registry import register
from ...spaces import ObsSpec
from ..layers import FastLinear
from .base import Encoder, conv_out_size, conv_stem
from .registry import ENCODERS

__all__ = ["ConvTransformerEncoder"]


@register(ENCODERS, "convtransformer")
class ConvTransformerEncoder(Encoder):
    """Conv stem -> patch tokens + frame/position embeddings -> transformer."""

    accepts = ("image",)

    def __init__(self, spec: ObsSpec, cfg: EncoderConfig) -> None:
        super().__init__(cfg.frames, cfg.feature_dim)
        self.check_spec(spec)
        channels, height, width = spec.shape
        self.stem = conv_stem(channels, cfg.cnn_channels, cfg.cnn_kernel, cfg.cnn_pool)
        self.n_tokens = conv_out_size(height, len(cfg.cnn_channels), cfg.cnn_pool) * (
            conv_out_size(width, len(cfg.cnn_channels), cfg.cnn_pool)
        )
        self.proj = nn.Conv2d(cfg.cnn_channels[-1], cfg.tf_d_model, 1)
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
