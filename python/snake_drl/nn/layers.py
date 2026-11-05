"""Layers with a measured fast path for CPU training.

``nn.Linear`` dispatches to ``addmm``.  On this torch build (2.15, aarch64 CPU)
``addmm`` is dramatically slower than ``matmul`` for exactly the shapes this
project uses -- measured with 4 threads:

======================  ==========  ==========
shape (M,K)x(K,N)       linear      matmul
======================  ==========  ==========
(16,64)x(64,64)          7.0 ms      0.045 ms
(256,64)x(64,64)         4.0 ms      0.051 ms
(256,256)x(256,128)      8.8 ms      0.122 ms
(1024,128)x(128,64)      2.8 ms      0.132 ms
(256,768)x(768,256)      6.7 ms      0.444 ms
======================  ==========  ==========

Fifteen to a hundred and fifty times, i.e. linear layers alone dominated a
policy forward.  :class:`FastLinear` keeps the ``nn.Linear`` parameter layout
(so state dicts and initialisation are unchanged) and only changes the forward
to the matmul form.  Set ``SNAKE_DRL_LINEAR=addmm`` to fall back to the stock
implementation on a platform where ``addmm`` is the faster path.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

_USE_MATMUL = os.environ.get("SNAKE_DRL_LINEAR", "matmul").lower() != "addmm"


class FastLinear(nn.Linear):
    """``nn.Linear`` whose forward uses ``matmul`` instead of ``addmm``."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if not _USE_MATMUL:
            return super().forward(input)
        output = torch.matmul(input, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output
