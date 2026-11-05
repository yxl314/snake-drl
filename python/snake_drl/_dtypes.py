"""The single mapping from :class:`~snake_drl.spaces.ObsSpec` dtypes to real ones.

Keeping it in one private module avoids every torch-side component repeating a
two-entry dictionary and, more importantly, keeps them from disagreeing.
"""

from __future__ import annotations

import numpy as np
import torch

from .spaces import ObsSpec

NUMERIC = {"uint8": np.uint8, "float32": np.float32}
TORCH = {"uint8": torch.uint8, "float32": torch.float32}


def numpy_dtype(spec: ObsSpec) -> type[np.generic]:
    return NUMERIC[spec.dtype]


def torch_dtype(spec: ObsSpec) -> torch.dtype:
    return TORCH[spec.dtype]
