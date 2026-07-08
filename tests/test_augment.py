"""Tests for the bounding-box augmentation helpers."""

from __future__ import annotations

import numpy as np

from wildlife_detector.data.augment import hflip_boxes


def test_hflip_xywhn_mirrors_x_center():
    out = hflip_boxes([[0.25, 0.5, 0.2, 0.3]])
    np.testing.assert_allclose(out[0], [0.75, 0.5, 0.2, 0.3])


def test_hflip_labeled_rows_preserve_class():
    out = hflip_boxes([[3, 0.2, 0.5, 0.1, 0.1]])
    assert out[0, 0] == 3  # class id untouched
    np.testing.assert_allclose(out[0, 1], 0.8)


def test_hflip_is_involution():
    boxes = np.array([[1, 0.3, 0.4, 0.2, 0.2], [0, 0.9, 0.1, 0.05, 0.05]])
    twice = hflip_boxes(hflip_boxes(boxes))
    np.testing.assert_allclose(twice, boxes)


def test_hflip_empty_is_safe():
    out = hflip_boxes(np.zeros((0, 5)))
    assert out.shape[0] == 0
