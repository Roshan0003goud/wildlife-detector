"""Drawing helpers for detections and the on-screen HUD.

OpenCV is imported lazily so that importing this module (e.g. for the colour
palette) does not require a working ``cv2`` install.
"""

from __future__ import annotations

import colorsys
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def color_palette(n: int) -> list[tuple[int, int, int]]:
    """Return ``n`` visually distinct BGR colours (deterministic per index)."""
    colors: list[tuple[int, int, int]] = []
    for i in range(max(n, 1)):
        hue = (i * 0.61803398875) % 1.0  # golden-ratio spacing for good spread
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR for OpenCV
    return colors


def draw_detections(
    frame: NDArray,
    boxes_xyxy: NDArray,
    class_ids: Sequence[int],
    scores: Sequence[float],
    names: Sequence[str],
    track_ids: Sequence[int] | None = None,
    palette: list[tuple[int, int, int]] | None = None,
    thickness: int = 2,
) -> NDArray:
    """Draw labelled boxes onto ``frame`` in place and return it.

    ``track_ids`` (when provided) are rendered as ``#id`` prefixes so tracked
    objects can be followed across frames.
    """
    import cv2  # local import: only needed when actually rendering

    palette = palette or color_palette(len(names))
    boxes = np.asarray(boxes_xyxy, dtype=int).reshape(-1, 4)

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        cls = int(class_ids[idx])
        color = palette[cls % len(palette)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        name = names[cls] if 0 <= cls < len(names) else str(cls)
        label = f"{name} {float(scores[idx]):.2f}"
        if track_ids is not None:
            label = f"#{int(track_ids[idx])} {label}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_hud(frame: NDArray, fps: float, count: int) -> NDArray:
    """Overlay a translucent header with live FPS and detection count."""
    import cv2

    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(
        frame,
        f"FPS: {fps:5.1f}   Animals: {count}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame
