"""Metric sinks: JSONL on disk, optional TensorBoard, human-readable console.

JSONL avoids the "columns change over time" problem of CSV and stays trivially
appendable; TensorBoard is imported lazily so it remains an optional dependency.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

__all__ = ["MetricLogger"]


class MetricLogger:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        console: bool = True,
        tensorboard: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "metrics.jsonl"
        self._file = self.path.open("a", encoding="utf-8")
        self._console = console
        self._tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(str(self.run_dir / "tb"))
            except ImportError:  # pragma: no cover - optional dependency
                print("tensorboard is not installed; continuing without it", file=sys.stderr)
        self._last = time.time()

    def log(self, step: int, metrics: dict[str, float]) -> None:
        record: dict[str, float | int | None] = {"step": int(step)}
        for key, value in metrics.items():
            number = float(value)
            # JSON has no NaN/Infinity; keep the line parseable by strict readers.
            record[key] = number if math.isfinite(number) else None
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        if self._tb is not None:
            for key, metric in record.items():
                if key != "step" and metric is not None:
                    self._tb.add_scalar(key, float(metric), step)
        if self._console:
            now = time.time()
            elapsed = now - self._last
            self._last = now
            body = "  ".join(
                f"{key}={metric:.4g}" if isinstance(metric, float) else f"{key}=n/a"
                for key, metric in record.items()
                if key != "step"
            )
            print(f"[{step:>9d}] {body}  ({elapsed:.1f}s)", flush=True)

    def close(self) -> None:
        self._file.close()
        if self._tb is not None:
            self._tb.close()

    def __enter__(self) -> MetricLogger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
