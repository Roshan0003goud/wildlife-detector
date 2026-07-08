"""Tests for bounding-box geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest

from wildlife_detector.utils.boxes import (
    box_iou,
    clip_boxes,
    xywhn_to_xyxy,
    xyxy_to_xywhn,
)


def test_xywhn_to_xyxy_centered_box():
    out = xywhn_to_xyxy([[0.5, 0.5, 0.5, 0.5]], 100, 100)
    np.testing.assert_allclose(out[0], [25, 25, 75, 75])


def test_xywhn_xyxy_roundtrip():
    boxes = np.array([[0.3, 0.4, 0.2, 0.1], [0.5, 0.5, 0.4, 0.4]])
    xyxy = xywhn_to_xyxy(boxes, 640, 480)
    back = xyxy_to_xywhn(xyxy, 640, 480)
    np.testing.assert_allclose(back, boxes, atol=1e-9)


def test_box_iou_identical_is_one():
    box = np.array([[0, 0, 10, 10]])
    assert box_iou(box, box)[0, 0] == pytest.approx(1.0)


def test_box_iou_disjoint_is_zero():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[20, 20, 30, 30]])
    assert box_iou(a, b)[0, 0] == pytest.approx(0.0)


def test_box_iou_half_overlap():
    a = np.array([[0, 0, 10, 10]])       # area 100
    b = np.array([[5, 0, 15, 10]])       # area 100, intersection 50
    # union = 150, iou = 50/150
    assert box_iou(a, b)[0, 0] == pytest.approx(50 / 150)


def test_box_iou_matrix_shape():
    a = np.zeros((3, 4))
    b = np.zeros((5, 4))
    assert box_iou(a, b).shape == (3, 5)


def test_clip_boxes_stays_in_bounds():
    out = clip_boxes([[-5, -5, 120, 90]], 100, 80)
    np.testing.assert_allclose(out[0], [0, 0, 100, 80])


def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        xywhn_to_xyxy([[1, 2, 3]], 10, 10)
