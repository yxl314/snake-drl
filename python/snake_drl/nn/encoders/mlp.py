"""Flattened multi-layer perceptron over the whole window."""

from __future__ import annotations

import torch
import torch.nn as nn

from ...config import EncoderConfig
from ...registry import register
from ...spaces import ObsSpec
from ..layers import FastLinear
from .base import Encoder
from .registry import ENCODERS

__all__ = ["MLPEncoder"]


@register(ENCODERS, "mlp")
class MLPEncoder(Encoder):
    """Flattens ``T`` frames into one vector and applies an MLP.

    Works for image and vector observations; with ``frames > 1`` it is plain
    frame stacking rather than a temporal model.
    """

    accepts = ("image", "vector")

    def __init__(self, spec: ObsSpec, cfg: EncoderConfig) -> None:
        super().__init__(cfg.frames, cfg.feature_dim)
        self.check_spec(spec)
        layers: list[nn.Module] = []
        prev = spec.frame_elems * cfg.frames
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
