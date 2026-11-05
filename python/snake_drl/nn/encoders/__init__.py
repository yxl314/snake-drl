"""The six interchangeable sequence encoders.

Every encoder consumes ``(B, T, *frame_shape)`` and returns ``(B, feature_dim)``.
Keeping the time axis inside the encoder is what lets PPO and the DQN family
share one policy/buffer contract: an algorithm always sees "one observation per
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
transformer variants are meant for a GPU or a small board.
"""

from __future__ import annotations

from ...config import EncoderConfig
from ...spaces import ObsSpec
from .base import Encoder
from .cnn import CNNEncoder
from .convgru import ConvGRUEncoder
from .convtransformer import ConvTransformerEncoder
from .gru import GRUEncoder
from .mlp import MLPEncoder
from .registry import ENCODERS
from .transformer import TransformerEncoder

__all__ = [
    "CNNEncoder",
    "ConvGRUEncoder",
    "ConvTransformerEncoder",
    "ENCODERS",
    "Encoder",
    "GRUEncoder",
    "MLPEncoder",
    "TransformerEncoder",
    "build_encoder",
]


def build_encoder(spec: ObsSpec, cfg: EncoderConfig) -> Encoder:
    """Instantiate the encoder named by ``cfg.type``."""
    if cfg.type not in ENCODERS:
        available = ", ".join(ENCODERS.names())
        raise ValueError(f"unknown encoder type {cfg.type!r}; available: {available}")
    return ENCODERS.get(cfg.type)(spec, cfg)  # type: ignore[return-value]
