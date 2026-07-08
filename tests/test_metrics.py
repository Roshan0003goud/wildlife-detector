"""Tests for the from-scratch detection metrics."""

from __future__ import annotations

import numpy as np
import pytest

from wildlife_detector.evaluation.metrics import (
    ImageGroundTruth,
    ImagePrediction,
    average_precision,
    evaluate_detections,
)

NAMES = ["a", "b"]


def test_average_precision_perfect_curve():
    recalls = np.array([0.5, 1.0])
    precisions = np.array([1.0, 1.0])
    assert average_precision(recalls, precisions) == pytest.approx(1.0)


def test_perfect_predictions_score_one():
    preds = [ImagePrediction(
        boxes=np.array([[10, 10, 50, 50]], float),
        scores=np.array([0.9]),
        classes=np.array([0]),
    )]
    gts = [ImageGroundTruth(boxes=np.array([[10, 10, 50, 50]], float), classes=np.array([0]))]
    m = evaluate_detections(preds, gts, NAMES)
    assert m.map50 == pytest.approx(1.0)
    assert m.precision == pytest.approx(1.0, abs=1e-6)
    assert m.recall == pytest.approx(1.0, abs=1e-6)
    assert m.mean_iou == pytest.approx(1.0, abs=1e-6)


def test_false_positive_lowers_precision():
    preds = [ImagePrediction(
        boxes=np.array([[10, 10, 50, 50], [100, 100, 120, 120]], float),
        scores=np.array([0.9, 0.8]),
        classes=np.array([0, 0]),
    )]
    gts = [ImageGroundTruth(boxes=np.array([[10, 10, 50, 50]], float), classes=np.array([0]))]
    m = evaluate_detections(preds, gts, NAMES)
    # One TP + one FP at report confidence => precision 0.5, recall 1.0.
    assert m.precision == pytest.approx(0.5, abs=1e-6)
    assert m.recall == pytest.approx(1.0, abs=1e-6)


def test_missed_detection_lowers_recall_and_map():
    preds = [ImagePrediction(
        boxes=np.array([[10, 10, 50, 50]], float),
        scores=np.array([0.9]),
        classes=np.array([0]),
    )]
    gts = [ImageGroundTruth(
        boxes=np.array([[10, 10, 50, 50], [100, 100, 140, 140]], float),
        classes=np.array([0, 0]),
    )]
    m = evaluate_detections(preds, gts, NAMES)
    assert m.recall == pytest.approx(0.5, abs=1e-6)
    assert m.map50 == pytest.approx(0.5, abs=1e-6)  # AP integrates to 0.5


def test_support_counts_only_present_classes():
    preds = [ImagePrediction(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), int))]
    gts = [ImageGroundTruth(np.array([[0, 0, 10, 10]], float), np.array([1]))]
    m = evaluate_detections(preds, gts, NAMES)
    assert m.support == {"a": 0, "b": 1}


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_detections([], [ImageGroundTruth(np.zeros((0, 4)), np.zeros((0,), int))], NAMES)
