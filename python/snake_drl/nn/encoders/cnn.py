"""Frame-stacked convolutional encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from ...config import EncoderConfig
from ...registry import register
from ...spaces import ObsSpec
from ..layers import FastLinear
from .base import Encoder, conv_out_size, conv_stem
from .registry import ENCODERS

__all__ = ["CNNEncoder"]


@register(ENCODERS, "cnn")
class CNNEncoder(Encoder):
    """Stacks the window on the channel axis and runs a small conv stack."""

    accepts = ("image",)

    def __init__(self, spec: ObsSpec, cfg: EncoderConfig) -> None:
        super().__init__(cfg.frames, cfg.feature_dim)
        self.check_spec(spec)
        channels, height, width = spec.shape
        self.body = conv_stem(channels * cfg.frames, cfg.cnn_channels, cfg.cnn_kernel, cfg.cnn_pool)
        out_h = min(2, conv_out_size(height, len(cfg.cnn_channels), cfg.cnn_pool))
        out_w = min(2, conv_out_size(width, len(cfg.cnn_channels), cfg.cnn_pool))
        self.pool = nn.AdaptiveAvgPool2d((out_h, out_w))
        flat = cfg.cnn_channels[-1] * out_h * out_w
        self.head = nn.Sequential(
            nn.Flatten(),
            FastLinear(flat, cfg.feature_dim),
            nn.LayerNorm(cfg.feature_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch, frames = obs.shape[:2]
        x = obs.reshape(batch, frames * obs.shape[2], *obs.shape[3:])
        return self.head(self.pool(self.body(x)))
