"""GRU over the frame window."""

from __future__ import annotations

import torch
import torch.nn as nn

from ...config import EncoderConfig
from ...registry import register
from ...spaces import ObsSpec
from ..layers import FastLinear
from .base import Encoder
from .registry import ENCODERS

__all__ = ["GRUEncoder"]


@register(ENCODERS, "gru")
class GRUEncoder(Encoder):
    """Per-frame linear embedding, then a GRU; the last hidden state is used.

    The hidden state starts at zero for every window, which is what keeps the
    encoder stateless from the algorithm's point of view.
    """

    accepts = ("image", "vector")

    def __init__(self, spec: ObsSpec, cfg: EncoderConfig) -> None:
        super().__init__(cfg.frames, cfg.feature_dim)
        self.check_spec(spec)
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
