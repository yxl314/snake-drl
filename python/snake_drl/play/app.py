"""PyQt6 human play mode for the snake core.

Run with::

    python -m snake_drl.play
    snake-play

Controls: arrow keys / WASD turn, space pauses, R restarts, +/- changes
speed, Esc quits.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .game import DOWN, LEFT, RIGHT, UP, GameSession

SPEEDS = (220, 180, 140, 110, 85, 65, 45)  # milliseconds per tick
DEFAULT_SPEED = 3

COLOR_BG = QColor("#0b1220")
COLOR_BOARD = QColor("#111c30")
COLOR_GRID = QColor("#1b2a45")
COLOR_BODY = QColor("#22c55e")
COLOR_BODY_EDGE = QColor("#15803d")
COLOR_HEAD = QColor("#a7f3d0")
COLOR_FOOD = QColor("#ef4444")
COLOR_TEXT = QColor("#e2e8f0")
COLOR_MUTED = QColor("#94a3b8")
COLOR_OVERLAY = QColor(5, 10, 20, 190)


class BoardView(QWidget):
    """Paints the (3,H,W) observation planes."""

    def __init__(self, session: GameSession, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self.overlay = ""
        self.setMinimumSize(420, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def show_overlay(self, text: str) -> None:
        if text != self.overlay:
            self.overlay = text
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), COLOR_BG)

        board = self.session.board
        height, width = int(board.shape[1]), int(board.shape[2])

        margin = 14.0
        cell = min(
            max(1.0, (self.width() - 2 * margin) / width),
            max(1.0, (self.height() - 2 * margin) / height),
        )
        board_w = cell * width
        board_h = cell * height
        x0 = (self.width() - board_w) / 2.0
        y0 = (self.height() - board_h) / 2.0

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(COLOR_BOARD)
        painter.drawRoundedRect(QRectF(x0, y0, board_w, board_h), 10.0, 10.0)

        pen = QPen(COLOR_GRID)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        for i in range(width + 1):
            x = x0 + i * cell
            painter.drawLine(QPointF(x, y0), QPointF(x, y0 + board_h))
        for i in range(height + 1):
            y = y0 + i * cell
            painter.drawLine(QPointF(x0, y), QPointF(x0 + board_w, y))

        painter.setPen(Qt.PenStyle.NoPen)
        inset = cell * 0.09

        for y, x in np.argwhere(board[0]):
            rect = QRectF(
                x0 + x * cell + inset,
                y0 + y * cell + inset,
                cell - 2 * inset,
                cell - 2 * inset,
            )
            painter.setBrush(COLOR_BODY)
            painter.setPen(QPen(COLOR_BODY_EDGE, max(1.0, cell * 0.04)))
            painter.drawRoundedRect(rect, cell * 0.22, cell * 0.22)

        for y, x in np.argwhere(board[2]):
            center = QPointF(x0 + (x + 0.5) * cell, y0 + (y + 0.5) * cell)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(COLOR_FOOD)
            painter.drawEllipse(center, cell * 0.3, cell * 0.3)
            painter.setBrush(QColor("#fca5a5"))
            painter.drawEllipse(
                QPointF(center.x() - cell * 0.09, center.y() - cell * 0.09),
                cell * 0.08,
                cell * 0.08,
            )

        for y, x in np.argwhere(board[1]):
            rect = QRectF(
                x0 + x * cell + inset * 0.7,
                y0 + y * cell + inset * 0.7,
                cell - 1.4 * inset,
                cell - 1.4 * inset,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(COLOR_HEAD)
            painter.drawRoundedRect(rect, cell * 0.28, cell * 0.28)

        if self.overlay:
            painter.fillRect(self.rect(), COLOR_OVERLAY)
            painter.setPen(COLOR_TEXT)
            font = painter.font()
            font.setPointSizeF(max(12.0, min(24.0, cell * 1.4)))
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                QRectF(0, 0, self.width(), self.height()),
                Qt.AlignmentFlag.AlignCenter,
                self.overlay,
            )


class GameWindow(QMainWindow):
    def __init__(self, session: GameSession | None = None, speed_index: int = DEFAULT_SPEED) -> None:
        super().__init__()
        self.session = session or GameSession()
        self.speed_index = max(0, min(len(SPEEDS) - 1, speed_index))
        self.started = False
        self.paused = False

        self.setWindowTitle("贪吃蛇 · snake-drl")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.board = BoardView(self.session)
        self.score_label = QLabel()
        self.score_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._style_label(self.score_label, 14, bold=True)

        self.hint_label = QLabel("方向键 / WASD 转向 · 空格 暂停 · R 重开 · +/- 调速 · Esc 退出")
        self._style_label(self.hint_label, 10, color=COLOR_MUTED)

        header = QHBoxLayout()
        header.setContentsMargins(4, 0, 4, 0)
        header.addWidget(self.score_label)
        header.addStretch(1)
        self.speed_label = QLabel()
        self._style_label(self.speed_label, 11, color=COLOR_MUTED)
        header.addWidget(self.speed_label)

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)
        layout.addLayout(header)
        layout.addWidget(self.board, 1)
        layout.addWidget(self.hint_label)

        central = QWidget()
        central.setLayout(layout)
        central.setStyleSheet("background-color: #0b1220;")
        self.setCentralWidget(central)
        self.setStyleSheet("QMainWindow { background-color: #0b1220; }")

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)

        self._refresh_hud()
        self._refresh_overlay()

    # ------------------------------------------------------------------ timer

    def _tick(self) -> None:
        if self.session.game_over:
            self.timer.stop()
            self._refresh_hud()
            self._refresh_overlay()
            return
        self.session.step()
        self._refresh_hud()
        self._refresh_overlay()
        self.board.update()
        if self.session.game_over:
            self.timer.stop()

    def _start(self) -> None:
        if self.started:
            return
        self.started = True
        self.paused = False
        self.timer.start(SPEEDS[self.speed_index])
        self._refresh_overlay()

    def _toggle_pause(self) -> None:
        if not self.started:
            self._start()
            return
        if self.session.game_over:
            return
        self.paused = not self.paused
        if self.paused:
            self.timer.stop()
        else:
            self.timer.start(SPEEDS[self.speed_index])
        self._refresh_overlay()

    def _restart(self) -> None:
        self.session.reset()
        self.timer.stop()
        self.started = False
        self.paused = False
        self._refresh_hud()
        self._refresh_overlay()
        self.board.update()

    def _change_speed(self, delta: int) -> None:
        new_index = max(0, min(len(SPEEDS) - 1, self.speed_index + delta))
        if new_index == self.speed_index:
            return
        self.speed_index = new_index
        if self.started and not self.paused and not self.session.game_over:
            self.timer.start(SPEEDS[self.speed_index])
        self._refresh_hud()

    # -------------------------------------------------------------------- hud

    def _refresh_hud(self) -> None:
        s = self.session
        self.score_label.setText(f"得分 {s.score}    长度 {s.length}    步数 {s.steps}")
        self.speed_label.setText(f"速度 {self.speed_index + 1}/{len(SPEEDS)}")

    def _refresh_overlay(self) -> None:
        if self.session.game_over:
            text = "你赢了！按 R 重新开始" if self.session.won else "游戏结束 · 按 R 重新开始"
        elif not self.started:
            text = "按方向键或空格开始"
        elif self.paused:
            text = "已暂停 · 空格继续"
        else:
            text = ""
        self.board.show_overlay(text)

    @staticmethod
    def _style_label(label: QLabel, point_size: int, *, bold: bool = False, color: QColor = COLOR_TEXT) -> None:
        font = QFont()
        font.setPointSize(point_size)
        font.setBold(bold)
        label.setFont(font)
        label.setStyleSheet(f"color: {color.name()};")

    # ------------------------------------------------------------------ input

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.isAutoRepeat():
            event.accept()
            return

        key = event.key()
        directions = {
            Qt.Key.Key_Up: UP,
            Qt.Key.Key_W: UP,
            Qt.Key.Key_Right: RIGHT,
            Qt.Key.Key_D: RIGHT,
            Qt.Key.Key_Down: DOWN,
            Qt.Key.Key_S: DOWN,
            Qt.Key.Key_Left: LEFT,
            Qt.Key.Key_A: LEFT,
        }

        if key in directions:
            if not self.started:
                self._start()
            if self.paused:
                self.paused = False
                self.timer.start(SPEEDS[self.speed_index])
                self._refresh_overlay()
            self.session.turn(directions[key])
            self.board.update()
            event.accept()
            return

        if key == Qt.Key.Key_Space:
            self._toggle_pause()
            event.accept()
            return

        if key == Qt.Key.Key_R:
            self._restart()
            event.accept()
            return

        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._change_speed(1)
            event.accept()
            return

        if key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self._change_speed(-1)
            event.accept()
            return

        if key == Qt.Key.Key_Escape:
            self.close()
            event.accept()
            return

        super().keyPressEvent(event)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play snake with the snake-drl core.")
    parser.add_argument("--width", type=int, default=16, help="board width (3-64)")
    parser.add_argument("--height", type=int, default=16, help="board height (3-64)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--speed",
        type=int,
        default=DEFAULT_SPEED + 1,
        choices=range(1, len(SPEEDS) + 1),
        metavar=f"1-{len(SPEEDS)}",
        help="initial speed level",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("snake-drl play")

    session = GameSession(width=args.width, height=args.height, seed=args.seed)
    window = GameWindow(session, speed_index=args.speed - 1)
    window.resize(620, 700)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
