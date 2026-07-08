"""Shared, dependency-light helpers: logging, box geometry, visualisation."""

from __future__ import annotations

from wildlife_detector.utils.boxes import (
    box_iou,
    clip_boxes,
    xywhn_to_xyxy,
    xyxy_to_xywhn,
)
from wildlife_detector.utils.logging import get_logger

__all__ = [
    "box_iou",
    "clip_boxes",
    "xywhn_to_xyxy",
    "xyxy_to_xywhn",
    "get_logger",
]
