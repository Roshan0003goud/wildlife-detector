"""Tests for the IoU-based object tracker."""

from __future__ import annotations

import numpy as np

from wildlife_detector.detection.tracker import IoUTracker


def test_same_object_keeps_id_across_frames():
    tracker = IoUTracker(iou_threshold=0.3)
    ids1 = tracker.update(np.array([[10, 10, 50, 50]]), np.array([0]))
    # Slightly moved box (still high IoU) should retain the id.
    ids2 = tracker.update(np.array([[12, 12, 52, 52]]), np.array([0]))
    assert ids1[0] == ids2[0]


def test_new_object_gets_new_id():
    tracker = IoUTracker(iou_threshold=0.3)
    tracker.update(np.array([[10, 10, 50, 50]]), np.array([0]))
    ids = tracker.update(
        np.array([[12, 12, 52, 52], [200, 200, 240, 240]]), np.array([0, 0])
    )
    assert ids[0] != ids[1]
    assert len(set(ids.tolist())) == 2


def test_disjoint_box_starts_fresh_track():
    tracker = IoUTracker(iou_threshold=0.3)
    ids1 = tracker.update(np.array([[10, 10, 50, 50]]), np.array([0]))
    ids2 = tracker.update(np.array([[300, 300, 340, 340]]), np.array([0]))
    assert ids1[0] != ids2[0]


def test_class_separation_prevents_id_switch():
    tracker = IoUTracker(iou_threshold=0.3, match_across_classes=False)
    ids1 = tracker.update(np.array([[10, 10, 50, 50]]), np.array([0]))
    # Same location but a different class => should NOT reuse the id.
    ids2 = tracker.update(np.array([[10, 10, 50, 50]]), np.array([1]))
    assert ids1[0] != ids2[0]


def test_track_expires_after_max_age():
    tracker = IoUTracker(iou_threshold=0.3, max_age=2)
    first = tracker.update(np.array([[10, 10, 50, 50]]), np.array([0]))
    for _ in range(3):  # exceed max_age with empty frames
        tracker.update(np.zeros((0, 4)), np.zeros((0,), int))
    revived = tracker.update(np.array([[10, 10, 50, 50]]), np.array([0]))
    assert first[0] != revived[0]  # old track was retired, new id assigned


def test_reset_clears_state():
    tracker = IoUTracker()
    tracker.update(np.array([[10, 10, 50, 50]]), np.array([0]))
    tracker.reset()
    ids = tracker.update(np.array([[10, 10, 50, 50]]), np.array([0]))
    assert ids[0] == 1
