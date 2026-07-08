"""Inference, object tracking and the OpenCV real-time video runner."""

from __future__ import annotations

from wildlife_detector.detection.detector import Detection, WildlifeDetector
from wildlife_detector.detection.tracker import IoUTracker
from wildlife_detector.detection.video import VideoDetector

__all__ = ["Detection", "WildlifeDetector", "IoUTracker", "VideoDetector"]
