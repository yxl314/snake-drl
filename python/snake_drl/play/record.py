"""Record a trained policy playing to an animated GIF.

The recorder is deliberately headless: the observation planes are rendered
straight to RGB frames and Pillow writes them out, so it runs over SSH, in CI or
inside a container where PyQt6 and a display do not exist. What it shows is what
the Qt window shows -- both read the same ``(3, H, W)`` planes from
:class:`~snake_drl.play.agent.AgentSession`.

The palette mirrors ``play/app.py``. A GIF in a slightly different shade would
still be a correct recording; the colours are duplicated so the recording path
stays free of Qt, not because they are load-bearing.

Run it from the CLI with::

    snake-drl play -m runs/<run>/latest.pt --record demo.gif
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:  # pragma: no cover - typing only; importing the agent pulls in torch
    from .agent import AgentSession

__all__ = ["COLORS", "render_frame", "record", "write_gif"]

# Same values as the Qt front-end (``play/app.py``): the GIF is meant to look
# like a recording of that window.
COLORS: dict[str, str] = {
    "board": "#111c30",
    "grid": "#1b2a45",
    "body": "#22c55e",
    "body_edge": "#15803d",
    "head": "#a7f3d0",
    "food": "#ef4444",
    "food_highlight": "#fca5a5",
}

DEFAULT_SCALE = 14  # pixels per board cell
TAIL_FRAMES = 5  # hold the death frame briefly, so an episode reads as one

# Below this many pixels per cell the grid lines dominate the snake.
_GRID_MIN_SCALE = 4


def render_frame(board: np.ndarray, scale: int = DEFAULT_SCALE) -> np.ndarray:
    """Render one ``(3, H, W)`` observation as an RGB image.

    Args:
        board: Body, head and food planes, as produced by
            ``AgentSession.board`` or ``GameSession.board``.
        scale: Pixels per board cell.

    Returns:
        np.ndarray: ``(H * scale, W * scale, 3)`` uint8 RGB image.

    Raises:
        ValueError: ``board`` is not ``(3, H, W)``, or ``scale < 1``.
    """
    planes = np.asarray(board)
    if planes.ndim != 3 or planes.shape[0] != 3:
        raise ValueError(f"expected a (3, H, W) board, got shape {planes.shape}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    height, width = int(planes.shape[1]), int(planes.shape[2])
    image = Image.new("RGB", (width * scale, height * scale), COLORS["board"])
    draw = ImageDraw.Draw(image)

    if scale >= _GRID_MIN_SCALE:
        for x in range(1, width):
            draw.line([(x * scale, 0), (x * scale, height * scale - 1)], fill=COLORS["grid"])
        for y in range(1, height):
            draw.line([(0, y * scale), (width * scale - 1, y * scale)], fill=COLORS["grid"])

    # Cell shapes follow app.py: body tiles with a darker edge, food as a dot
    # with a highlight, and the head drawn last so it sits on top of the body.
    inset = scale * 0.09
    for y, x in np.argwhere(planes[0]):
        draw.rounded_rectangle(
            (
                x * scale + inset,
                y * scale + inset,
                (x + 1) * scale - inset,
                (y + 1) * scale - inset,
            ),
            radius=scale * 0.22,
            fill=COLORS["body"],
            outline=COLORS["body_edge"],
            width=max(1, int(scale * 0.06)),
        )

    radius = scale * 0.3
    for y, x in np.argwhere(planes[2]):
        center_x = (x + 0.5) * scale
        center_y = (y + 0.5) * scale
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            fill=COLORS["food"],
        )
        highlight = max(1.0, scale * 0.08)
        draw.ellipse(
            (
                center_x - scale * 0.09 - highlight,
                center_y - scale * 0.09 - highlight,
                center_x - scale * 0.09 + highlight,
                center_y - scale * 0.09 + highlight,
            ),
            fill=COLORS["food_highlight"],
        )

    head_inset = scale * 0.06
    for y, x in np.argwhere(planes[1]):
        draw.rounded_rectangle(
            (
                x * scale + head_inset,
                y * scale + head_inset,
                (x + 1) * scale - head_inset,
                (y + 1) * scale - head_inset,
            ),
            radius=scale * 0.28,
            fill=COLORS["head"],
        )

    return np.asarray(image, dtype=np.uint8)


def write_gif(frames: list[np.ndarray], out_path: str | Path, *, fps: int = 12) -> Path:
    """Write RGB frames as an endlessly looping GIF.

    Args:
        frames: RGB arrays of identical size.
        out_path: Destination file; parent directories are created.
        fps: Frames per second, used for the per-frame delay.

    Returns:
        Path: The written file.

    Raises:
        ValueError: ``fps < 1``, no frames were given, or frame sizes differ.
    """
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}")
    if not frames:
        raise ValueError("cannot write a GIF without frames")
    images = [
        frame if isinstance(frame, Image.Image)
        else _to_gif_frame(np.asarray(frame, dtype=np.uint8))
        for frame in frames
    ]
    size = images[0].size
    for index, image in enumerate(images):
        if image.size != size:
            raise ValueError(f"frame {index} is {image.size}, expected {size}")

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = images
    # GIF delays are centiseconds; integer division is what the format can
    # represent, and the max(1, ...) keeps absurdly high fps values valid.
    first.save(
        destination,
        save_all=True,
        append_images=rest,
        duration=max(1, 1000 // fps),
        loop=0,
    )
    return destination


def _to_gif_frame(rgb: np.ndarray) -> Image.Image:
    """Convert one rendered RGB frame to a palettised image.

    A GIF is 256 colours per frame, so keeping the RGB arrays alive until the
    end of the rollout wastes two thirds of the memory. Long episodes (a good
    agent can survive thousands of steps) is exactly when that matters.
    """
    return Image.fromarray(rgb, mode="RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)


def _rollout(session: AgentSession, *, episodes: int, scale: int) -> list[Image.Image]:
    """Play ``episodes`` complete games and render every frame in order."""
    frames: list[Image.Image] = []
    for _ in range(episodes):
        session.reset()
        frames.append(_to_gif_frame(render_frame(session.board, scale)))
        while not session.game_over:
            session.step()
            frames.append(_to_gif_frame(render_frame(session.board, scale)))
        # A death that lasts exactly one GIF frame is unreadable, and the pause
        # also marks where one episode ends and the next begins.
        frames.extend([frames[-1]] * TAIL_FRAMES)
    return frames


def record(
    checkpoint: str | Path,
    out_path: str | Path,
    *,
    episodes: int = 3,
    fps: int = 12,
    scale: int = DEFAULT_SCALE,
) -> Path:
    """Record a checkpoint's greedy policy to a GIF.

    Each episode runs until the engine reports the game over; the terminal
    frame is then held for :data:`TAIL_FRAMES` frames.

    Args:
        checkpoint: Path to a ``.pt`` checkpoint written by the trainer.
        out_path: Destination GIF.
        episodes: Number of complete episodes to record (>= 1).
        fps: Playback speed in frames per second (>= 1).
        scale: Pixels per board cell (>= 1).

    Returns:
        Path: The written GIF.

    Raises:
        ValueError: An argument is out of range.
        FileNotFoundError: The checkpoint does not exist.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    # Imported here rather than at module level: recording needs torch, while
    # the frame rendering (and its tests) must not.
    from .agent import AgentSession

    session = AgentSession(str(checkpoint))
    try:
        frames = _rollout(session, episodes=episodes, scale=scale)
    finally:
        session.close()
    return write_gif(frames, out_path, fps=fps)
