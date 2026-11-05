"""The single mapping from :class:`~snake_drl.spaces.ObsSpec` dtypes to torch dtypes.

Keeping it in one private module avoids every torch-side component repeating a
two-entry dictionary and, more importantly, keeps them from disagreeing.
"""

from __future__ import annotations

import torch

from .spaces import ObsSpec

TORCH = {"uint8": torch.uint8, "float32": torch.float32}


def torch_dtype(spec: ObsSpec) -> torch.dtype:
    return TORCH[spec.dtype]
