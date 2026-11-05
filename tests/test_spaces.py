from __future__ import annotations

import pytest

from snake_drl.spaces import ActionSpec, ObsSpec


def test_obs_spec_image_and_vector():
    image = ObsSpec((3, 8, 8))
    assert image.frame_elems == 3 * 8 * 8
    assert image.windowed(4) == (4, 3, 8, 8)
    vector = ObsSpec((5,), kind="vector", dtype="float32")
    assert vector.frame_elems == 5


def test_obs_spec_rejects_bad_shapes():
    with pytest.raises(ValueError):
        ObsSpec((3, 8))
    with pytest.raises(ValueError):
        ObsSpec((5, 1), kind="vector")
    with pytest.raises(ValueError):
        ObsSpec((0, 8, 8))
    with pytest.raises(ValueError):
        ObsSpec((3, 8, 8), kind="audio")


def test_action_spec():
    assert ActionSpec(3).n == 3
    with pytest.raises(ValueError):
        ActionSpec(0)
    with pytest.raises(ValueError):
        ActionSpec(2, kind="continuous")
