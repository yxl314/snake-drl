"""GIF 录制模块的单元测试：逐帧渲染、GIF 写出、参数校验。

这些用例不加载 checkpoint、也不 import torch：渲染与写 GIF 是纯 numpy + PIL；
整条 ``record()`` 链路用注入的假 agent 模块验证，不依赖训练产物。
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PIL import Image, ImageColor
from snake_drl.play.record import (
    COLORS,
    TAIL_FRAMES,
    _rollout,
    record,
    render_frame,
    write_gif,
)


def _rgb(name: str) -> tuple[int, int, int]:
    """把调色板常量转成 PIL 可比较的 RGB 三元组。"""
    return ImageColor.getrgb(COLORS[name])


class _FakeSession:
    """两 tick 就结束的一局，board 每步都变，便于检查帧序列。"""

    def __init__(self) -> None:
        self.board = np.zeros((3, 4, 4), dtype=np.uint8)
        self.game_over = False
        self.steps = 0
        self.resets = 0
        self.closed = False

    def reset(self) -> None:
        self.resets += 1
        self.steps = 0
        self.game_over = False
        self.board = np.zeros((3, 4, 4), dtype=np.uint8)
        self.board[1, 0, 0] = 1

    def step(self) -> bool:
        self.steps += 1
        self.board = np.zeros((3, 4, 4), dtype=np.uint8)
        self.board[0, 0, self.steps] = 1
        self.game_over = self.steps >= 2
        return True

    def close(self) -> None:
        self.closed = True


def test_render_frame_shape_and_palette():
    """body/head/food 三平面应画成与 app.py 一致的配色，尺寸为 scale 倍。"""
    board = np.zeros((3, 5, 7), dtype=np.uint8)
    board[0, 2, 1:4] = 1  # 蛇身
    board[1, 2, 3] = 1  # 蛇头（引擎同时把它记在 body 平面上）
    board[2, 4, 5] = 1  # 食物
    scale = 10

    frame = render_frame(board, scale)

    assert frame.shape == (5 * scale, 7 * scale, 3)
    assert frame.dtype == np.uint8
    assert tuple(frame[0, 0]) == _rgb("board")  # 空格子：棋盘底色
    assert tuple(frame[2 * scale + 5, 1 * scale + 5]) == _rgb("body")
    assert tuple(frame[2 * scale + 5, 3 * scale + 5]) == _rgb("head")
    assert tuple(frame[4 * scale + 5, 5 * scale + 5]) == _rgb("food")
    # 网格线画在格子边界上，不影响格子中心。
    assert tuple(frame[0, scale]) == _rgb("grid")


def test_render_frame_handles_the_smallest_scale():
    """scale=1 时不应报错（网格线省略），只是每格一个像素。"""
    board = np.zeros((3, 5, 7), dtype=np.uint8)
    frame = render_frame(board, 1)
    assert frame.shape == (5, 7, 3)


def test_render_frame_validates_input():
    """形状不是 (3,H,W) 或 scale < 1 时应抛 ValueError。"""
    with pytest.raises(ValueError, match=r"\(3, H, W\)"):
        render_frame(np.zeros((2, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="scale"):
        render_frame(np.zeros((3, 4, 4), dtype=np.uint8), 0)


def test_write_gif_roundtrip(tmp_path):
    """写出的 GIF 应能被 Pillow 重新打开：尺寸、帧数、循环与帧间隔都对。"""
    frames = []
    for red in (10, 120, 250):
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[:, :, 0] = red
        frames.append(frame)

    out = write_gif(frames, tmp_path / "nested" / "demo.gif", fps=10)

    assert out == tmp_path / "nested" / "demo.gif"
    assert out.is_file() and out.stat().st_size > 0
    with Image.open(out) as gif:
        assert gif.format == "GIF"
        assert gif.size == (16, 12)
        assert gif.n_frames == 3
        assert gif.info.get("loop") == 0
        durations = []
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(gif.info.get("duration"))
        assert durations == [100, 100, 100]


def test_write_gif_validates_arguments(tmp_path):
    """fps < 1、没有帧、帧尺寸不一致都应抛 ValueError。"""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="fps"):
        write_gif([frame], tmp_path / "fps.gif", fps=0)
    with pytest.raises(ValueError, match="without frames"):
        write_gif([], tmp_path / "empty.gif")
    with pytest.raises(ValueError, match="expected"):
        write_gif([frame, np.zeros((5, 4, 3), dtype=np.uint8)], tmp_path / "bad.gif")


def test_rollout_plays_every_episode_to_the_end_and_holds_the_last_frame():
    """每局从 reset 跑到 game_over，并在最后重复 TAIL_FRAMES 帧静帧。"""
    session = _FakeSession()

    frames = _rollout(session, episodes=2, scale=4)

    per_episode = 3 + TAIL_FRAMES  # 初始帧 + 2 步死亡 + 静帧
    assert len(frames) == 2 * per_episode
    assert session.resets == 2
    for episode in range(2):
        base = episode * per_episode
        death_frame = frames[base + 2]
        tail = frames[base + 3 : base + per_episode]
        assert len(tail) == TAIL_FRAMES
        assert all(np.array_equal(frame, death_frame) for frame in tail)
    assert session.closed is False  # close() 由 record() 负责，_rollout 不管资源


def test_record_validates_arguments_before_touching_the_checkpoint(tmp_path):
    """参数非法时应先抛 ValueError，而不是先去加载 checkpoint。"""
    with pytest.raises(ValueError, match="episodes"):
        record(tmp_path / "missing.pt", tmp_path / "x.gif", episodes=0)
    with pytest.raises(ValueError, match="fps"):
        record(tmp_path / "missing.pt", tmp_path / "x.gif", fps=0)
    with pytest.raises(ValueError, match="scale"):
        record(tmp_path / "missing.pt", tmp_path / "x.gif", scale=0)


def test_record_writes_a_gif_without_a_real_checkpoint(tmp_path, monkeypatch):
    """record() 延迟导入 AgentSession；注入假模块即可端到端验证，不依赖 torch/权重。"""
    created_checkpoints: list[str] = []

    class FakeAgentSession(_FakeSession):
        def __init__(self, checkpoint: str, *args, **kwargs) -> None:
            created_checkpoints.append(checkpoint)
            super().__init__()

    fake_agent = types.ModuleType("snake_drl.play.agent")
    fake_agent.AgentSession = FakeAgentSession
    monkeypatch.setitem(sys.modules, "snake_drl.play.agent", fake_agent)

    # record() 先检查文件存在，再延迟导入 AgentSession；这里给个占位文件即可。
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"")
    out = record(checkpoint, tmp_path / "demo.gif", episodes=2, fps=20, scale=4)

    assert created_checkpoints == [str(checkpoint)]
    assert out.is_file() and out.stat().st_size > 0
    with Image.open(out) as gif:
        assert gif.format == "GIF"
        assert gif.size == (16, 16)  # 4x4 格 * scale 4
        # Pillow 会把连续相同帧（静帧）合并成一个图像块并累加时长，
        # 所以用总播放时长而不是 n_frames 验证所有帧都写进去了。
        durations = []
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(gif.info.get("duration"))
        expected_frames = 2 * (3 + TAIL_FRAMES)
        assert sum(durations) == expected_frames * (1000 // 20)
