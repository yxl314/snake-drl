"""训练报告模块的单元测试：metrics.jsonl 解析、摘要表对齐、曲线 PNG。

matplotlib 是可选依赖：需要它的用例在缺失时 skip，纯文本摘要部分始终可测。
"""

from __future__ import annotations

import json
import sys

import pytest
from snake_drl.report import (
    _running_max,
    _smooth,
    format_summary,
    load_metrics,
    plot_report,
    resolve_metrics_path,
    summarize,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _write_metrics(path, records):
    """把若干条记录写成 JSONL，返回文件路径。"""
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


def test_resolve_metrics_path_accepts_run_dir_or_file(tmp_path):
    """运行目录与 metrics.jsonl 本身都应能定位到同一个文件。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = _write_metrics(run_dir / "metrics.jsonl", [{"step": 0, "loss": 1.0}])

    assert resolve_metrics_path(run_dir) == metrics
    assert resolve_metrics_path(metrics) == metrics
    with pytest.raises(FileNotFoundError, match="metrics.jsonl"):
        resolve_metrics_path(tmp_path / "missing-run")


def test_load_metrics_skips_null_values_and_blank_lines(tmp_path):
    """null 值应整条丢弃（而不是变成 NaN），空行应被忽略。"""
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"step": 0, "loss": None, "entropy": 0.5}),
                "",
                json.dumps({"step": 1, "loss": 1.5}),
                "   ",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_metrics(path)

    assert [row["step"] for row in rows] == [0.0, 1.0]
    assert "loss" not in rows[0]  # 丢弃的是键，不是改成 NaN
    assert rows[0]["entropy"] == 0.5
    assert rows[1]["loss"] == 1.5


def test_load_metrics_rejects_non_object_lines(tmp_path):
    """合法 JSON 但不是对象（例如数组）应报出带行号的 ValueError。"""
    path = tmp_path / "metrics.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_metrics(path)


def test_summarize_first_last_best_mean_and_count():
    """摘要统计按每个指标实际出现的记录计算，step 不进入摘要。"""
    rows = [
        {"step": 0, "train/episode_return": 1.0},
        {"step": 10, "train/episode_return": 3.0},
        {"step": 20, "train/episode_return": 2.0},
        {"step": 30, "other": 5.0},
    ]

    summary = summarize(rows)

    stats = summary["train/episode_return"]
    assert stats["first"] == 1.0
    assert stats["last"] == 2.0
    assert stats["best"] == 3.0
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["count"] == 3.0
    assert "step" not in summary
    assert summary["other"]["count"] == 1.0
    assert summarize([]) == {}


def test_smooth_is_a_trailing_mean_that_keeps_the_length():
    """滑动平均是尾随窗口，输出与输入等长；窗口大于数据时退化为累计均值。"""
    assert _smooth([1.0, 2.0, 3.0, 4.0], 2) == [1.0, 1.5, 2.5, 3.5]
    assert _smooth([1.0, 2.0], 7) == [1.0, 1.5]
    assert _smooth([], 3) == []


def test_running_max_is_monotone_and_starts_from_the_first_value():
    """运行最大值回答“是否曾超过 X”，不能被后来的低点拉低。"""
    assert _running_max([1.0, 3.0, 2.0, 5.0, 4.0]) == [1.0, 3.0, 3.0, 5.0, 5.0]
    assert _running_max([]) == []


def test_format_summary_columns_align_and_headline_keys_come_first():
    """摘要表列宽对齐，且高信号指标排在前面、其余指标按名字排在后面。"""
    rows = [
        {"step": 0, "loss": 1.0, "train/episode_return": 2.0, "zzz": 0.5},
        {"step": 1, "loss": 3.0, "train/episode_return": 4.0, "zzz": 0.7},
    ]

    table = format_summary(summarize(rows))
    lines = table.splitlines()

    # 表头 + 三个指标 + 一行方向说明（loss 类指标的 best 取最小值）
    assert len(lines) == 5
    assert lines[0].startswith("metric")
    body = lines[:-1]
    assert {len(line) for line in body} == {len(body[0])}, "每一行的列位置必须一致"
    assert lines[1].startswith("train/episode_return")
    assert lines[2].startswith("loss")
    assert lines[3].startswith("zzz")
    assert "minimum for loss-like metrics" in lines[-1]
    assert format_summary({}) == ""


def test_best_follows_metric_direction():
    """loss 类指标的 best 是最小值，回报类指标是最大值。"""
    rows = [
        {"step": 0, "loss": 5.0, "train/episode_return": 0.1},
        {"step": 1, "loss": 1.0, "train/episode_return": 0.9},
    ]
    summary = summarize(rows)
    assert summary["loss"]["best"] == 1.0
    assert summary["train/episode_return"]["best"] == 0.9


def test_plot_report_writes_a_png(tmp_path):
    """有 matplotlib 时，四面板曲线应写出一张可解码的 PNG。"""
    pytest.importorskip("matplotlib")
    rows = [
        {
            "step": index * 10,
            "train/episode_return": 0.1 * index,
            "train/episode_length": 5.0 + index,
            "loss": 1.0 / (index + 1),
            "entropy": 0.5,
        }
        for index in range(5)
    ]

    out = plot_report(rows, tmp_path / "nested" / "report.png", title="unit-test")

    assert out == tmp_path / "nested" / "report.png"
    assert out.is_file() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == PNG_MAGIC


def test_plot_report_rejects_unplottable_rows(tmp_path):
    """空数据或没有任何已知指标时应在写文件之前报错。"""
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="no records"):
        plot_report([], tmp_path / "empty.png")
    with pytest.raises(ValueError, match="no known metric keys"):
        plot_report([{"step": 0, "custom": 1.0}], tmp_path / "unknown.png")


def test_plot_report_without_matplotlib_raises_import_error(tmp_path, monkeypatch):
    """matplotlib 缺失时应抛 ImportError（提示安装方式），而不是崩在别处。"""
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ImportError, match="matplotlib is required"):
        plot_report([{"step": 0, "loss": 1.0}], tmp_path / "chart.png")


def test_panels_declare_only_known_keys():
    """面板引用的指标键必须是训练真的会写出来的键。

    面板里写错一个键名的后果是"这一格永远空白"，不会报错——所以用清单锁住。
    """
    from snake_drl.report import PANELS

    known = {
        "train/episode_return",
        "train/episode_length",
        "eval/return",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "q_mean",
    }
    for _title, keys in PANELS:
        for key in keys:
            assert key in known, f"panel references unknown metric {key!r}"
