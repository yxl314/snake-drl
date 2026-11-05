"""Encoder base class and the conv-shape helpers shared by the CNN variants."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn

from ...spaces import ObsSpec

__all__ = ["Encoder", "group_norm", "conv_out_size", "conv_stem"]


def group_norm(channels: int) -> nn.Module:
    """GroupNorm with eight groups when possible, one otherwise.

    GroupNorm (rather than BatchNorm) keeps the normalisation independent of the
    batch composition, which matters for a replay buffer that mixes old and new
    policies.
    """
    return nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)


def conv_out_size(size: int, layers: int, stride: int) -> int:
    """Output size of ``layers`` 3x3 stride-``stride`` convs with padding 1."""
    for _ in range(layers):
        size = math.ceil(size / stride)
    return max(1, size)


def conv_stem(channels: int, widths: Iterable[int], kernel: int, stride: int) -> nn.Sequential:
    """Stack of conv/GroupNorm/SiLU blocks turning ``channels`` into ``widths[-1]``."""
    blocks: list[nn.Module] = []
    prev = channels
    for width in widths:
        blocks += [
            nn.Conv2d(prev, width, kernel, stride=stride, padding=1),
            group_norm(width),
            nn.SiLU(),
        ]
        prev = width
    return nn.Sequential(*blocks)


class Encoder(nn.Module):
    """Maps ``(B, T, *frame)`` to ``(B, feature_dim)``.

    ``frames`` is a property of the encoder, not of the algorithm: the runner
    keeps exactly this many frames per environment and every algorithm sees one
    observation per environment per step.
    """

    frames: int
    feature_dim: int
    accepts: tuple[str, ...] = ("image", "vector")

    def __init__(self, frames: int, feature_dim: int) -> None:
        super().__init__()
        self.frames = int(frames)
        self.feature_dim = int(feature_dim)

    def check_spec(self, spec: ObsSpec) -> None:
        if spec.kind not in self.accepts:
            name = type(self).__name__
            raise ValueError(f"{name} accepts {self.accepts} observations, got {spec.kind!r}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError
