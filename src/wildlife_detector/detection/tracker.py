"""A lightweight IoU-based multi-object tracker.

This is a deliberately small, dependency-free tracker (in the spirit of SORT,
minus the Kalman filter) that assigns stable integer ids to detections across
frames so animals can be *counted* and *followed* in a live stream rather than
re-detected anew every frame. Matching is pure NumPy and fully unit-tested.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wildlife_detector.utils.boxes import box_iou


class _Track:
    __slots__ = ("id", "box", "class_id", "hits", "age", "time_since_update")

    def __init__(self, track_id: int, box: NDArray, class_id: int) -> None:
        self.id = track_id
        self.box = box
        self.class_id = class_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0


class IoUTracker:
    """Greedy IoU tracker.

    Parameters
    ----------
    iou_threshold:
        Minimum IoU for a detection to be associated with an existing track.
    max_age:
        Number of consecutive frames a track may go unmatched before it is
        dropped (handles brief occlusions).
    match_across_classes:
        If ``False`` (default) a detection can only extend a track of the same
        class, which prevents id-switches between, say, a fox and a deer.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        match_across_classes: bool = False,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1].")
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.match_across_classes = match_across_classes
        self._tracks: list[_Track] = []
        self._next_id = 1

    def reset(self) -> None:
        """Forget all tracks (e.g. when starting a new video)."""
        self._tracks.clear()
        self._next_id = 1

    @property
    def num_active_tracks(self) -> int:
        return sum(1 for t in self._tracks if t.time_since_update == 0)

    def update(self, boxes_xyxy: NDArray, class_ids: NDArray | None = None) -> NDArray:
        """Advance the tracker by one frame.

        Returns an integer array of track ids, one per input box, in the same
        order as ``boxes_xyxy``.
        """
        boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        n = len(boxes)
        if class_ids is None:
            class_ids = np.full(n, -1, dtype=int)
        else:
            class_ids = np.asarray(class_ids, dtype=int).reshape(-1)

        # Every existing track gets one frame older this step.
        for track in self._tracks:
            track.age += 1
            track.time_since_update += 1

        assigned = np.full(n, -1, dtype=int)

        if self._tracks and n:
            assigned = self._match(boxes, class_ids, assigned)

        # Unmatched detections spawn new tracks.
        for det_idx in range(n):
            if assigned[det_idx] == -1:
                track = _Track(self._next_id, boxes[det_idx], int(class_ids[det_idx]))
                self._next_id += 1
                self._tracks.append(track)
                assigned[det_idx] = track.id

        # Retire tracks that have been unmatched for too long.
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]
        return assigned

    def _match(self, boxes: NDArray, class_ids: NDArray, assigned: NDArray) -> NDArray:
        track_boxes = np.array([t.box for t in self._tracks], dtype=np.float64)
        iou = box_iou(track_boxes, boxes)  # shape (num_tracks, num_dets)

        if not self.match_across_classes:
            track_cls = np.array([t.class_id for t in self._tracks])[:, None]
            iou = np.where(track_cls == class_ids[None, :], iou, 0.0)

        # Greedily consume the highest-IoU pairs first.
        order = np.argsort(iou, axis=None)[::-1]
        pairs = np.column_stack(np.unravel_index(order, iou.shape))
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        for track_idx, det_idx in pairs:
            track_idx, det_idx = int(track_idx), int(det_idx)
            if iou[track_idx, det_idx] < self.iou_threshold:
                break  # remaining pairs are all below threshold
            if track_idx in matched_tracks or det_idx in matched_dets:
                continue
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)
            track = self._tracks[track_idx]
            track.box = boxes[det_idx]
            track.class_id = int(class_ids[det_idx])
            track.time_since_update = 0
            track.hits += 1
            assigned[det_idx] = track.id
        return assigned
