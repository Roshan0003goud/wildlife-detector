"""Vectorised bounding-box geometry.

Pure NumPy so it is fast, fully unit-testable, and importable without torch or
OpenCV. Boxes are expressed in two conventions:

* ``xywhn`` - normalised ``[x_center, y_center, width, height]`` in ``[0, 1]``
  (the YOLO label format).
* ``xyxy``  - absolute pixel ``[x_min, y_min, x_max, y_max]``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_2d(boxes: NDArray) -> FloatArray:
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[-1] != 4:
        raise ValueError(f"Expected boxes of shape (N, 4); got {arr.shape}.")
    return arr


def xywhn_to_xyxy(boxes: NDArray, img_w: int, img_h: int) -> FloatArray:
    """Convert normalised centre boxes to absolute pixel corner boxes."""
    b = _as_2d(boxes)
    cx, cy, w, h = b[:, 0] * img_w, b[:, 1] * img_h, b[:, 2] * img_w, b[:, 3] * img_h
    out = np.empty_like(b)
    out[:, 0] = cx - w / 2.0
    out[:, 1] = cy - h / 2.0
    out[:, 2] = cx + w / 2.0
    out[:, 3] = cy + h / 2.0
    return out


def xyxy_to_xywhn(boxes: NDArray, img_w: int, img_h: int) -> FloatArray:
    """Convert absolute pixel corner boxes to normalised centre boxes."""
    b = _as_2d(boxes)
    w = (b[:, 2] - b[:, 0]) / img_w
    h = (b[:, 3] - b[:, 1]) / img_h
    out = np.empty_like(b)
    out[:, 0] = (b[:, 0] / img_w) + w / 2.0
    out[:, 1] = (b[:, 1] / img_h) + h / 2.0
    out[:, 2] = w
    out[:, 3] = h
    return out


def clip_boxes(boxes: NDArray, img_w: int, img_h: int) -> FloatArray:
    """Clamp ``xyxy`` boxes so they stay inside the image bounds."""
    b = _as_2d(boxes).copy()
    b[:, 0::2] = b[:, 0::2].clip(0, img_w)
    b[:, 1::2] = b[:, 1::2].clip(0, img_h)
    return b


def box_area(boxes: NDArray) -> FloatArray:
    b = _as_2d(boxes)
    return (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)


def box_iou(boxes_a: NDArray, boxes_b: NDArray) -> FloatArray:
    """Pairwise IoU matrix between two sets of ``xyxy`` boxes.

    Returns an ``(len(a), len(b))`` array where entry ``[i, j]`` is the
    Intersection-over-Union of ``boxes_a[i]`` and ``boxes_b[j]``.
    """
    a = _as_2d(boxes_a)
    b = _as_2d(boxes_b)

    area_a = box_area(a)[:, None]
    area_b = box_area(b)[None, :]

    # Intersection rectangle corners via broadcasting.
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clip(0)
    inter_h = (inter_y2 - inter_y1).clip(0)
    intersection = inter_w * inter_h

    union = area_a + area_b - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / union, 0.0)
    return iou.astype(np.float64)
