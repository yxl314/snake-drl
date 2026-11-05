"""Convolutional GRU: a 2-D hidden state carried across the window."""

from __future__ import annotations

import torch
import torch.nn as nn

from ...config import EncoderConfig
from ...registry import register
from ...spaces import ObsSpec
from ..layers import FastLinear
from .base import Encoder, conv_stem
from .registry import ENCODERS

__all__ = ["ConvGRUEncoder"]


@register(ENCODERS, "convgru")
class ConvGRUEncoder(Encoder):
    """Conv stem followed by proper GRU gates on spatial feature maps.

    The gates are computed separately for the input and the previous state
    (``r = sigmoid(gx_r + gh_r)``, ``n = tanh(gx_n + r * gh_n)``), matching the
    usual GRU formulation rather than a simplified single-conv variant.
    """

    accepts = ("image",)

    def __init__(self, spec: ObsSpec, cfg: EncoderConfig) -> None:
        super().__init__(cfg.frames, cfg.feature_dim)
        self.check_spec(spec)
        self.stem = conv_stem(spec.shape[0], cfg.cnn_channels, cfg.cnn_kernel, cfg.cnn_pool)
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
            reset = torch.sigmoid(gx[:, :h] + gh[:, :h])
            update = torch.sigmoid(gx[:, h : 2 * h] + gh[:, h : 2 * h])
            candidate = torch.tanh(gx[:, 2 * h :] + reset * gh[:, 2 * h :])
            state = (1.0 - update) * candidate + update * state
        return self.head(state.mean(dim=(2, 3)))
