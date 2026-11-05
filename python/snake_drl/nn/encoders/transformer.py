"""Transformer over the frame window."""

from __future__ import annotations

import torch
import torch.nn as nn

from ...config import EncoderConfig
from ...registry import register
from ...spaces import ObsSpec
from ..layers import FastLinear
from .base import Encoder
from .registry import ENCODERS

__all__ = ["TransformerEncoder"]


@register(ENCODERS, "transformer")
class TransformerEncoder(Encoder):
    """Per-frame embedding + learned positional encoding, last token as output."""

    accepts = ("image", "vector")

    def __init__(self, spec: ObsSpec, cfg: EncoderConfig) -> None:
        super().__init__(cfg.frames, cfg.feature_dim)
        self.check_spec(spec)
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
        # The nested-tensor fast path is slower for the short windows used here.
        self.body = nn.TransformerEncoder(layer, cfg.tf_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.tf_d_model), FastLinear(cfg.tf_d_model, cfg.feature_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch, frames = obs.shape[:2]
        z = self.inp(obs.reshape(batch, frames, -1)) + self.pos[:, :frames]
        return self.head(self.body(z)[:, -1])
