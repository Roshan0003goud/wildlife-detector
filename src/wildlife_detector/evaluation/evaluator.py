"""Run a trained detector over a dataset split and produce a metrics report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wildlife_detector.detection.detector import WildlifeDetector
from wildlife_detector.evaluation.metrics import (
    DetectionMetrics,
    ImageGroundTruth,
    ImagePrediction,
    evaluate_detections,
)
from wildlife_detector.utils.boxes import xywhn_to_xyxy
from wildlife_detector.utils.logging import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Evaluator:
    """Evaluate a checkpoint against a YOLO dataset split."""

    def __init__(
        self,
        weights: str | Path,
        dataset_yaml: str | Path,
        conf: float = 0.001,
        iou: float = 0.6,
        report_conf: float = 0.25,
        iou_threshold: float = 0.5,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        self.dataset_yaml = Path(dataset_yaml)
        self.report_conf = report_conf
        self.iou_threshold = iou_threshold
        self.detector = WildlifeDetector(
            weights, conf=conf, iou=iou, imgsz=imgsz, device=device
        )

    def run(self, split: str = "val", output_dir: str | Path = "runs/eval") -> DetectionMetrics:
        descriptor = self._load_descriptor()
        class_names = self._class_names(descriptor)
        image_paths = self._list_images(descriptor, split)
        logger.info("Evaluating %d %s images over %d classes",
                    len(image_paths), split, len(class_names))

        predictions: list[ImagePrediction] = []
        ground_truths: list[ImageGroundTruth] = []
        for img_path in image_paths:
            width, height = self._image_size(img_path)
            predictions.append(self._predict(img_path))
            ground_truths.append(self._ground_truth(img_path, width, height))

        metrics = evaluate_detections(
            predictions, ground_truths, class_names,
            iou_threshold=self.iou_threshold, report_conf=self.report_conf,
        )
        self._save_report(metrics, output_dir, split)
        self._log_report(metrics)
        return metrics

    # -- helpers ----------------------------------------------------------- #
    def _load_descriptor(self) -> dict[str, Any]:
        if not self.dataset_yaml.is_file():
            raise FileNotFoundError(f"Dataset descriptor not found: {self.dataset_yaml}")
        with self.dataset_yaml.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    @staticmethod
    def _class_names(descriptor: dict[str, Any]) -> list[str]:
        names = descriptor.get("names")
        if isinstance(names, dict):
            return [names[i] for i in sorted(names)]
        if isinstance(names, list):
            return list(names)
        raise ValueError("dataset.yaml must define `names`.")

    def _list_images(self, descriptor: dict[str, Any], split: str) -> list[Path]:
        base = Path(descriptor.get("path", self.dataset_yaml.parent))
        rel = descriptor.get(split)
        if rel is None:
            raise ValueError(f"Split {split!r} not present in {self.dataset_yaml}.")
        split_dir = (base / rel).resolve()
        images = [p for p in sorted(split_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            raise FileNotFoundError(f"No images found for split {split!r} in {split_dir}.")
        return images

    @staticmethod
    def _image_size(img_path: Path) -> tuple[int, int]:
        from PIL import Image  # lazy: only needed to read dimensions

        with Image.open(img_path) as im:
            return im.size  # (width, height)

    def _predict(self, img_path: Path) -> ImagePrediction:
        dets = self.detector.predict(str(img_path))
        if not dets:
            return ImagePrediction(
                boxes=np.zeros((0, 4)), scores=np.zeros((0,)), classes=np.zeros((0,), int)
            )
        return ImagePrediction(
            boxes=np.array([d.box for d in dets], dtype=np.float64),
            scores=np.array([d.score for d in dets], dtype=np.float64),
            classes=np.array([d.class_id for d in dets], dtype=int),
        )

    @staticmethod
    def _label_path(img_path: Path) -> Path:
        # YOLO convention: mirror .../images/... -> .../labels/... , .txt extension.
        parts = list(img_path.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                parts[i] = "labels"
                break
        return Path(*parts).with_suffix(".txt")

    def _ground_truth(self, img_path: Path, width: int, height: int) -> ImageGroundTruth:
        label_path = self._label_path(img_path)
        if not label_path.is_file():
            return ImageGroundTruth(boxes=np.zeros((0, 4)), classes=np.zeros((0,), int))
        rows = [
            [float(v) for v in line.split()]
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            return ImageGroundTruth(boxes=np.zeros((0, 4)), classes=np.zeros((0,), int))
        arr = np.asarray(rows, dtype=np.float64)
        classes = arr[:, 0].astype(int)
        boxes = xywhn_to_xyxy(arr[:, 1:5], width, height)
        return ImageGroundTruth(boxes=boxes, classes=classes)

    def _save_report(self, metrics: DetectionMetrics, output_dir: str | Path, split: str) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / f"metrics_{split}.json"
        report_path.write_text(json.dumps(metrics.as_dict(), indent=2), encoding="utf-8")
        logger.info("Saved evaluation report to %s", report_path)
        return report_path

    def _log_report(self, metrics: DetectionMetrics) -> None:
        logger.info("=" * 52)
        logger.info(" Evaluation summary")
        logger.info("-" * 52)
        logger.info("  mAP@0.5   : %.4f", metrics.map50)
        logger.info("  Precision : %.4f", metrics.precision)
        logger.info("  Recall    : %.4f", metrics.recall)
        logger.info("  F1        : %.4f", metrics.f1)
        logger.info("  Mean IoU  : %.4f", metrics.mean_iou)
        logger.info("-" * 52)
        logger.info("  %-14s %8s %8s", "class", "AP@0.5", "support")
        for name, ap in metrics.per_class_ap.items():
            logger.info("  %-14s %8.4f %8d", name, ap, metrics.support.get(name, 0))
        logger.info("=" * 52)
