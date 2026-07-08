"""Detection metrics (mAP / IoU / precision) and the evaluation runner."""

from __future__ import annotations

from wildlife_detector.evaluation.evaluator import Evaluator
from wildlife_detector.evaluation.metrics import (
    DetectionMetrics,
    ImageGroundTruth,
    ImagePrediction,
    average_precision,
    evaluate_detections,
)

__all__ = [
    "Evaluator",
    "DetectionMetrics",
    "ImageGroundTruth",
    "ImagePrediction",
    "average_precision",
    "evaluate_detections",
]
