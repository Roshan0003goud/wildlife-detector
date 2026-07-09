"""High-level inference wrapper around a trained YOLOv5 model.

:class:`WildlifeDetector` hides the ultralytics API behind a small, typed
surface: give it a frame (a file path or a raw BGR ``ndarray`` from OpenCV) and
it returns a list of :class:`Detection` objects. The model is loaded lazily on
first use so constructing the detector is cheap.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wildlife_detector.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """A single detected object in one frame."""

    box: tuple[float, float, float, float]  # xyxy, absolute pixels
    score: float
    class_id: int
    class_name: str
    track_id: int | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.box

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class WildlifeDetector:
    """Load a YOLOv5 checkpoint and run detection on images or video frames."""

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        self.weights = str(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self._model: Any | None = None
        self._names: dict[int, str] | None = None

    # -- lifecycle --------------------------------------------------------- #
    def load(self) -> None:
        """Eagerly load the underlying model (otherwise done on first predict)."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Could not import the inference backend (ultralytics/torch). "
                f"Underlying import error: {exc!r}"
            ) from exc
        if not Path(self.weights).is_file():
            # A bare model alias (e.g. "yolov5su.pt") is auto-downloaded by
            # ultralytics — this is what lets the hosted demo run without a
            # trained checkpoint. Only a genuine *path* that is missing errors.
            looks_like_path = os.sep in self.weights or (
                os.altsep is not None and os.altsep in self.weights
            )
            if looks_like_path:
                raise FileNotFoundError(
                    f"Model weights not found: {self.weights}. Train a model first or "
                    f"pass a valid --weights path."
                )
            logger.info(
                "No local file '%s'; loading it as a pretrained model (auto-download).",
                self.weights,
            )
        else:
            logger.info("Loading detector weights: %s", self.weights)
        self._model = YOLO(self.weights)
        self._names = dict(self._model.names)

    @property
    def names(self) -> dict[int, str]:
        self.load()
        assert self._names is not None
        return self._names

    @property
    def class_names(self) -> list[str]:
        names = self.names
        return [names[i] for i in range(len(names))]

    # -- inference --------------------------------------------------------- #
    def predict(self, frame: str | Path | NDArray) -> list[Detection]:
        """Run detection on a single image/frame and return detections."""
        self.load()
        assert self._model is not None
        results = self._model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        return self._parse(results[0])

    def _parse(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        names = self.names
        detections: list[Detection] = []
        for (x1, y1, x2, y2), score, cls in zip(xyxy, confs, clss):
            detections.append(
                Detection(
                    box=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(score),
                    class_id=int(cls),
                    class_name=names.get(int(cls), str(cls)),
                )
            )
        return detections


def detections_to_arrays(
    detections: Sequence[Detection],
) -> tuple[NDArray, NDArray, NDArray]:
    """Split a list of detections into ``(boxes_xyxy, class_ids, scores)`` arrays."""
    if not detections:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=np.float64),
        )
    boxes = np.array([d.box for d in detections], dtype=np.float64)
    class_ids = np.array([d.class_id for d in detections], dtype=int)
    scores = np.array([d.score for d in detections], dtype=np.float64)
    return boxes, class_ids, scores
