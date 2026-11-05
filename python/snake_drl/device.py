"""Device selection and process-wide performance knobs.

The engine is CPU-only and runs its own thread pool; torch defaults to one
intra-op thread per core.  Both at full width oversubscribes the machine --
measured on an 8-vCPU aarch64 box, ``torch.set_num_threads(8)`` made a 196 KiB
``copy_`` ~60x slower than with 4 threads (one straggler thread stalls every
OpenMP barrier), and end-to-end steps ~2x slower.  The defaults here are
therefore deliberately conservative and tunable through ``run.torch_threads``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

__all__ = ["Runtime", "auto_torch_threads", "configure_runtime", "resolve_device"]

_MPS = getattr(torch.backends, "mps", None)


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda``/``cuda:N``/``mps`` to a concrete device."""
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _MPS is not None and _MPS.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {name!r} requested but CUDA is not available")
    if device.type == "mps" and not (_MPS is not None and _MPS.is_available()):
        raise RuntimeError(f"device {name!r} requested but MPS is not available")
    return device


def auto_torch_threads() -> int:
    """Conservative intra-op thread count.

    Half the cores (capped at 8) leaves room for the engine's worker threads and
    for the caller.  Measured: 4 on an 8-core box beats both 1 and 8.
    """
    cores = os.cpu_count() or 4
    return max(1, min(8, cores // 2))


@dataclass(frozen=True)
class Runtime:
    """Process-wide settings resolved once at start-up."""

    device: torch.device
    torch_threads: int


def configure_runtime(device: torch.device, torch_threads: int = 0) -> Runtime:
    """Apply process-wide settings once, before any model or worker is built."""
    threads = int(torch_threads) if torch_threads else auto_torch_threads()
    torch.set_num_threads(threads)
    try:
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
    except RuntimeError:  # parallel work already started
        pass
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device.type == "cpu":
        torch.set_float32_matmul_precision("highest")
    return Runtime(device=device, torch_threads=threads)
