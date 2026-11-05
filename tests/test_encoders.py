from __future__ import annotations

import pytest
import torch

from snake_drl.config import EncoderConfig
from snake_drl.nn import ENCODERS, build_encoder
from snake_drl.spaces import ObsSpec

IMAGE = ObsSpec((3, 16, 16))
SMALL_IMAGE = ObsSpec((3, 8, 8))
VECTOR = ObsSpec((7,), kind="vector", dtype="float32")

IMAGE_ENCODERS = ("mlp", "cnn", "gru", "transformer", "convgru", "convtransformer")
VECTOR_ENCODERS = ("mlp", "gru", "transformer")


def _cfg(kind: str, frames: int) -> EncoderConfig:
    return EncoderConfig(
        type=kind,
        frames=frames,
        feature_dim=32,
        hidden_dims=[32, 32],
        cnn_channels=[8, 8],
        gru_hidden=16,
        tf_d_model=16,
        tf_heads=2,
        tf_layers=1,
        tf_ff=32,
    )


@pytest.mark.parametrize("kind", IMAGE_ENCODERS)
@pytest.mark.parametrize("frames", [1, 3])
def test_image_encoders(kind: str, frames: int):
    if kind in ("gru", "transformer", "convgru", "convtransformer") and frames == 1:
        pytest.skip("temporal encoders require frames >= 2")
    encoder = build_encoder(IMAGE, _cfg(kind, frames))
    obs = torch.rand(5, frames, 3, 16, 16)
    out = encoder(obs)
    assert out.shape == (5, 32)
    assert torch.isfinite(out).all()
    assert encoder.frames == frames


@pytest.mark.parametrize("kind", VECTOR_ENCODERS)
def test_vector_encoders(kind: str):
    encoder = build_encoder(VECTOR, _cfg(kind, 3))
    out = encoder(torch.rand(4, 3, 7))
    assert out.shape == (4, 32)


@pytest.mark.parametrize("kind", ("cnn", "convgru", "convtransformer"))
def test_image_only_encoders_reject_vectors(kind: str):
    with pytest.raises(ValueError, match="accepts"):
        build_encoder(VECTOR, _cfg(kind, 2))


def test_odd_sizes_and_gradients():
    spec = ObsSpec((3, 7, 5))
    for kind in ("cnn",):
        encoder = build_encoder(spec, _cfg(kind, 1))
        obs = torch.rand(2, 1, 3, 7, 5, requires_grad=False)
        out = encoder(obs)
        out.sum().backward()
        grads = [p.grad for p in encoder.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)


def test_registry_contents():
    assert set(ENCODERS.names()) == set(IMAGE_ENCODERS)
