"""Object-detection metrics implemented from scratch in NumPy.

Everything the evaluation report needs — Average Precision per class, mAP@0.5,
precision / recall / F1 at a reporting confidence, and mean IoU of matched
boxes — is computed here without depending on the training framework, which
keeps the numbers auditable and the module fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from wildlife_detector.utils.boxes import box_iou

_EPS = 1e-12


@dataclass
class ImagePrediction:
    """Detector output for one image."""

    boxes: NDArray  # (N, 4) xyxy
    scores: NDArray  # (N,)
    classes: NDArray  # (N,)


@dataclass
class ImageGroundTruth:
    """Ground-truth annotations for one image."""

    boxes: NDArray  # (M, 4) xyxy
    classes: NDArray  # (M,)


@dataclass
class DetectionMetrics:
    """Aggregate evaluation results."""

    map50: float
    precision: float
    recall: float
    f1: float
    mean_iou: float
    per_class_ap: dict[str, float] = field(default_factory=dict)
    support: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "mAP@0.5": round(self.map50, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mean_iou": round(self.mean_iou, 4),
            "per_class_ap": {k: round(v, 4) for k, v in self.per_class_ap.items()},
            "support": self.support,
        }


def average_precision(recalls: NDArray, precisions: NDArray) -> float:
    """All-points (VOC2010/COCO-style) AP from a precision-recall curve."""
    mrec = np.concatenate(([0.0], np.asarray(recalls, dtype=np.float64), [1.0]))
    mpre = np.concatenate(([0.0], np.asarray(precisions, dtype=np.float64), [0.0]))
    # Make precision monotonically decreasing (envelope).
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _ap_for_class(
    predictions: list[tuple[float, int, NDArray]],
    gt_by_image: dict[int, NDArray],
    n_gt: int,
    iou_threshold: float,
) -> float:
    """AP for a single class given all its predictions and ground truths."""
    if n_gt == 0:
        return 0.0
    if not predictions:
        return 0.0

    predictions = sorted(predictions, key=lambda p: p[0], reverse=True)
    matched: dict[int, NDArray] = {
        img: np.zeros(len(boxes), dtype=bool) for img, boxes in gt_by_image.items()
    }
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))

    for i, (_score, img_id, box) in enumerate(predictions):
        gt_boxes = gt_by_image.get(img_id)
        if gt_boxes is None or len(gt_boxes) == 0:
            fp[i] = 1
            continue
        ious = box_iou(box.reshape(1, 4), gt_boxes)[0]
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold and not matched[img_id][best]:
            tp[i] = 1
            matched[img_id][best] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / (n_gt + _EPS)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, _EPS)
    return average_precision(recalls, precisions)


def _greedy_match(
    pred: ImagePrediction, gt: ImageGroundTruth, iou_threshold: float, conf: float
) -> tuple[int, int, int, list[float]]:
    """Greedy per-image matching at a confidence threshold.

    Returns ``(tp, fp, fn, matched_ious)`` aggregated over all classes in the
    image — used for the human-readable precision/recall/IoU report.
    """
    keep = pred.scores >= conf
    p_boxes, p_cls, p_scores = pred.boxes[keep], pred.classes[keep], pred.scores[keep]
    order = np.argsort(p_scores)[::-1]
    p_boxes, p_cls = p_boxes[order], p_cls[order]

    gt_used = np.zeros(len(gt.boxes), dtype=bool)
    tp = 0
    matched_ious: list[float] = []

    for box, cls in zip(p_boxes, p_cls):
        candidates = np.where((gt.classes == cls) & (~gt_used))[0]
        if len(candidates) == 0:
            continue
        ious = box_iou(box.reshape(1, 4), gt.boxes[candidates])[0]
        best_local = int(np.argmax(ious))
        if ious[best_local] >= iou_threshold:
            tp += 1
            gt_used[candidates[best_local]] = True
            matched_ious.append(float(ious[best_local]))

    n_pred = len(p_boxes)
    n_gt = len(gt.boxes)
    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn, matched_ious


def evaluate_detections(
    predictions: list[ImagePrediction],
    ground_truths: list[ImageGroundTruth],
    class_names: list[str],
    iou_threshold: float = 0.5,
    report_conf: float = 0.25,
) -> DetectionMetrics:
    """Compute mAP@iou plus precision/recall/F1/mean-IoU at ``report_conf``."""
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must be the same length.")

    num_classes = len(class_names)

    # ---- mAP: gather predictions & gts per class across all images ---- #
    per_class_preds: list[list[tuple[float, int, NDArray]]] = [[] for _ in range(num_classes)]
    per_class_gt: list[dict[int, NDArray]] = [{} for _ in range(num_classes)]
    support = np.zeros(num_classes, dtype=int)

    for img_id, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        for box, score, cls in zip(pred.boxes, pred.scores, pred.classes):
            if 0 <= int(cls) < num_classes:
                per_class_preds[int(cls)].append((float(score), img_id, np.asarray(box)))
        for cls in range(num_classes):
            mask = gt.classes == cls
            if np.any(mask):
                per_class_gt[cls][img_id] = gt.boxes[mask]
                support[cls] += int(np.sum(mask))

    per_class_ap: dict[str, float] = {}
    for cls in range(num_classes):
        ap = _ap_for_class(
            per_class_preds[cls], per_class_gt[cls], int(support[cls]), iou_threshold
        )
        per_class_ap[class_names[cls]] = ap

    present = support > 0
    if present.any():
        map50 = float(
            np.mean([per_class_ap[class_names[c]] for c in range(num_classes) if present[c]])
        )
    else:
        map50 = 0.0

    # ---- precision / recall / F1 / mean IoU at the report confidence ---- #
    total_tp = total_fp = total_fn = 0
    all_ious: list[float] = []
    for pred, gt in zip(predictions, ground_truths):
        tp, fp, fn, ious = _greedy_match(pred, gt, iou_threshold, report_conf)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)

    precision = total_tp / (total_tp + total_fp + _EPS)
    recall = total_tp / (total_tp + total_fn + _EPS)
    f1 = 2 * precision * recall / (precision + recall + _EPS)
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    return DetectionMetrics(
        map50=map50,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        mean_iou=mean_iou,
        per_class_ap=per_class_ap,
        support={class_names[c]: int(support[c]) for c in range(num_classes)},
    )
