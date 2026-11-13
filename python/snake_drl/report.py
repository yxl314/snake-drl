"""Training report: turn ``metrics.jsonl`` into a summary and a chart.

Two outputs, one data source:

* a text summary printed to the console -- always available;
* a PNG figure with the essential curves -- needs matplotlib, which is an
  optional dependency (``pip install -e ".[plot]"``).

The figure is deliberately small: four panels covering what a training run is
actually judged by (return, episode length, loss, exploration/optimisation
diagnostics). Anything else belongs in the raw JSONL, which is always there.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "PANELS",
    "configure_fonts",
    "load_metrics",
    "plot_report",
    "resolve_metrics_path",
    "summarize",
]

# Rolling window (in logged records) for the smoothed training curve. Records
# arrive at ``train.log_every`` intervals, so this is "the last few logs".
SMOOTH_WINDOW = 7

# Panel layout: title -> candidate metric keys, in priority order. Only keys
# present in the run are drawn, so the same layout works for PPO and DQN.
PANELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("episode return", ("train/episode_return", "eval/return")),
    ("episode length", ("train/episode_length",)),
    ("loss", ("loss", "policy_loss", "value_loss")),
    ("diagnostics", ("entropy", "q_mean", "approx_kl")),
)

# Metrics where a smaller value is better. The summary reports "best" as the
# optimum in the metric's own direction: max return, min loss. Without this the
# table calls the worst loss the best one.
_LOWER_IS_BETTER = frozenset(
    ("loss", "policy_loss", "value_loss", "td_abs_mean", "approx_kl")
)

# Key whose running maximum is drawn as a dashed line: it answers "did we ever
# get better than X" without eyeballing the noisy curve. Only the training
# curve gets one -- the evaluation points are sparse enough to read directly,
# and a second dashed line just clutters the panel.
RUNNING_MAX_KEYS = ("train/episode_return",)


def resolve_metrics_path(target: str | Path) -> Path:
    """Accept a run directory or a ``metrics.jsonl`` path.

    Args:
        target: Run directory (contains ``metrics.jsonl``) or that file itself.

    Returns:
        Path: The metrics file.

    Raises:
        FileNotFoundError: Neither the file nor ``target/metrics.jsonl`` exists.
    """
    path = Path(target)
    if path.is_dir():
        path = path / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no metrics.jsonl at {target}")
    return path


def load_metrics(path: str | Path) -> list[dict[str, float]]:
    """Read a JSONL metrics file.

    Values written as ``null`` (non-finite numbers) are dropped rather than
    turned into NaN, so a single bad record cannot poison a whole curve.

    Args:
        path: Metrics file, or a run directory containing one.

    Returns:
        list[dict[str, float]]: Records in file order.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: A line is not a JSON object.
    """
    rows: list[dict[str, float]] = []
    with resolve_metrics_path(path).open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {lineno}: expected a JSON object")
            row = {key: float(value) for key, value in record.items() if value is not None}
            rows.append(row)
    return rows


def _series(rows: Sequence[dict[str, float]], key: str) -> tuple[list[float], list[float]]:
    """Extract ``(x, y)`` for one key, skipping records that lack it."""
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if key in row and "step" in row:
            xs.append(row["step"])
            ys.append(row[key])
    return xs, ys


def _smooth(values: Sequence[float], window: int) -> list[float]:
    """Trailing mean over ``window`` samples; the curve keeps its length."""
    out: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        out.append(total / min(index + 1, window))
    return out


def _running_max(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    best = -math.inf
    for value in values:
        best = max(best, value)
        out.append(best)
    return out


def _fmt(value: float) -> str:
    """Format a metric for the aligned summary table (never wider than 10)."""
    if not math.isfinite(value):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4g}"


def summarize(rows: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Per-key ``first``/``last``/``best``/``mean`` over the run.

    Args:
        rows: Records from :func:`load_metrics`.

    Returns:
        dict[str, dict[str, float]]: Empty for an empty input.
    """
    keys: set[str] = set()
    for row in rows:
        keys.update(row)
    keys.discard("step")
    summary: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        values = [row[key] for row in rows if key in row]
        if not values:
            continue
        summary[key] = {
            "first": values[0],
            "last": values[-1],
            "best": min(values) if key in _LOWER_IS_BETTER else max(values),
            "mean": sum(values) / len(values),
            "count": float(len(values)),
        }
    return summary


def configure_fonts() -> None:
    """Pick a font that can render CJK titles, without adding a dependency.

    Without this, Chinese panel titles render as boxes on most Linux installs
    and the figure silently becomes half-useless. Matplotlib is imported here
    rather than at module level so the text summary works without it.
    """
    import matplotlib
    from matplotlib import font_manager

    candidates = (
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    # Minus signs come from the CJK font's glyph set too.
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_report(
    rows: Sequence[dict[str, float]],
    out_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 150,
) -> Path:
    """Draw the training report and write it to ``out_path``.

    Args:
        rows: Records from :func:`load_metrics`.
        out_path: Destination image. The suffix decides the format.
        title: Figure title; defaults to the output file's parent directory.
        dpi: Raster resolution.

    Returns:
        Path: The written file.

    Raises:
        ImportError: matplotlib is not installed.
        ValueError: There is nothing to plot.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: a report must not need a display
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "matplotlib is required for charts; install it with "
            'pip install -e ".[plot]" or use --no-chart'
        ) from exc

    if not rows:
        raise ValueError("metrics file has no records to plot")
    configure_fonts()

    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    flat = list(axes.flat)
    drawn = 0
    for ax, (panel_title, keys) in zip(flat, PANELS, strict=False):
        plotted = 0
        for key in keys:
            xs, ys = _series(rows, key)
            if not xs:
                continue
            is_eval = key.startswith("eval/")
            # Evaluation points are sparse: draw them as markers so they read as
            # checkpoints rather than as a second, jagged training curve.
            # A single record draws no visible line; give short series markers.
            marker = "o" if (is_eval or len(xs) <= 2) else None
            if is_eval:
                ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, label=key)
            else:
                ax.plot(xs, _smooth(ys, SMOOTH_WINDOW), marker=marker,
                        markersize=4, linewidth=1.6, label=key)
            plotted += 1
            if key in RUNNING_MAX_KEYS:
                ax.plot(
                    xs,
                    _running_max(ys),
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    label=f"{key} (max)",
                )
        if not plotted:
            ax.set_visible(False)
            continue
        drawn += 1
        ax.set_title(panel_title)
        ax.set_xlabel("environment steps")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False)

    if drawn == 0:
        plt.close(figure)
        raise ValueError("metrics file has no known metric keys to plot")

    figure.suptitle(title or str(Path(out_path).parent), fontsize=12)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)
    return destination


# Highest-signal keys, in the order a person reads them. Anything else in the
# file is still summarised, just below these.
_HEADLINE_KEYS = (
    "train/episode_return",
    "eval/return",
    "train/episode_length",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "q_mean",
)


def format_summary(summary: dict[str, dict[str, float]]) -> str:
    """Render :func:`summarize` as an aligned text table.

    Args:
        summary: Output of :func:`summarize`.

    Returns:
        str: Multi-line table, or an empty string for an empty summary.
    """
    if not summary:
        return ""
    ordered = [key for key in _HEADLINE_KEYS if key in summary]
    ordered += [key for key in summary if key not in ordered]
    width = max(len(key) for key in ordered)
    lines = [
        f"{'metric':<{width}}  {'first':>10}  {'last':>10}  {'best':>10}  {'mean':>10}"
    ]
    for key in ordered:
        stats = summary[key]
        lines.append(
            f"{key:<{width}}  {_fmt(stats['first']):>10}  {_fmt(stats['last']):>10}  "
            f"{_fmt(stats['best']):>10}  {_fmt(stats['mean']):>10}"
        )
    if any(key in _LOWER_IS_BETTER for key in ordered):
        lines.append("best = minimum for loss-like metrics, maximum otherwise")
    return "\n".join(lines)
