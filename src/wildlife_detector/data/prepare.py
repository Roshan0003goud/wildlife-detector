"""Preprocessing pipeline: validate → split → materialise a YOLO dataset.

The pipeline that "reduced experimentation time" lives here: one command turns a
folder of raw images + YOLO labels into a reproducible, validated, train/val/test
dataset with a ready-to-train ``dataset.yaml``. The heavy lifting (splitting and
label validation) is written as pure functions so it is fast and unit-testable.
"""

from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from wildlife_detector.config import DataConfig
from wildlife_detector.utils.logging import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


class LabelError(ValueError):
    """Raised when a YOLO label file is malformed or references a bad class id."""


@dataclass(frozen=True)
class Pair:
    """An image path paired with its (possibly empty) label path."""

    image: Path
    label: Path


# --------------------------------------------------------------------------- #
# Pure, unit-testable helpers
# --------------------------------------------------------------------------- #
def validate_label_text(
    text: str, num_classes: int
) -> list[tuple[int, float, float, float, float]]:
    """Parse and validate the contents of a YOLO label file.

    Returns the list of parsed ``(class_id, cx, cy, w, h)`` rows. Raises
    :class:`LabelError` describing the first problem encountered. An empty file
    (a negative/background sample) is valid and yields an empty list.
    """
    rows: list[tuple[int, float, float, float, float]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise LabelError(f"line {lineno}: expected 5 values, got {len(parts)}: {line!r}")
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError as exc:
            raise LabelError(f"line {lineno}: non-numeric value: {line!r}") from exc

        if not 0 <= cls < num_classes:
            raise LabelError(f"line {lineno}: class id {cls} outside [0, {num_classes - 1}]")
        for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not 0.0 <= val <= 1.0:
                raise LabelError(
                    f"line {lineno}: {name}={val} not in [0, 1] (labels must be normalised)"
                )
        if w == 0 or h == 0:
            raise LabelError(f"line {lineno}: zero-area box {line!r}")
        rows.append((cls, cx, cy, w, h))
    return rows


def split_indices(
    n: int, ratios: tuple[float, float, float], seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Deterministically partition ``range(n)`` into train/val/test index lists.

    The ``val`` and ``test`` counts are rounded and the remainder goes to
    ``train`` so every sample is used exactly once.
    """
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    _, val_r, test_r = ratios
    n_val = round(n * val_r)
    n_test = round(n * test_r)
    n_val = min(n_val, n)
    n_test = min(n_test, n - n_val)

    val = indices[:n_val]
    test = indices[n_val : n_val + n_test]
    train = indices[n_val + n_test :]
    return train, val, test


def find_pairs(raw_dir: Path) -> list[Pair]:
    """Discover image/label pairs under ``raw_dir/images`` and ``raw_dir/labels``.

    Images without a matching label file are treated as background samples and
    paired with a non-existent label path (an empty label at train time).
    """
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Expected images directory at {images_dir}")

    pairs: list[Pair] = []
    for img in sorted(images_dir.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = labels_dir / f"{img.stem}.txt"
        pairs.append(Pair(image=img, label=label))
    return pairs


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class DatasetPreparer:
    """Turn a raw annotated folder into a validated, split YOLO dataset."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def run(self) -> Path:
        """Execute the full pipeline and return the path to ``dataset.yaml``."""
        cfg = self.config
        pairs = find_pairs(cfg.raw_dir)
        if not pairs:
            raise FileNotFoundError(f"No images found under {cfg.raw_dir / 'images'}")
        logger.info("Discovered %d images under %s", len(pairs), cfg.raw_dir)

        valid_pairs, total_boxes = self._validate(pairs)
        logger.info(
            "Validated %d/%d images (%d bounding boxes) across %d classes",
            len(valid_pairs),
            len(pairs),
            total_boxes,
            cfg.num_classes,
        )

        ratios = (cfg.split.train, cfg.split.val, cfg.split.test)
        train_idx, val_idx, test_idx = split_indices(len(valid_pairs), ratios, cfg.seed)
        assignment = {
            "train": [valid_pairs[i] for i in train_idx],
            "val": [valid_pairs[i] for i in val_idx],
            "test": [valid_pairs[i] for i in test_idx],
        }
        for split, items in assignment.items():
            logger.info("  %-5s split: %d images", split, len(items))

        self._materialise(assignment)
        yaml_path = self._write_dataset_yaml(assignment)
        logger.info("Dataset descriptor written to %s", yaml_path)
        return yaml_path

    # -- internal steps ---------------------------------------------------- #
    def _validate(self, pairs: list[Pair]) -> tuple[list[Pair], int]:
        valid: list[Pair] = []
        total_boxes = 0
        for pair in pairs:
            if pair.label.is_file():
                text = pair.label.read_text(encoding="utf-8")
                try:
                    rows = validate_label_text(text, self.config.num_classes)
                except LabelError as exc:
                    logger.warning("Skipping %s: %s", pair.image.name, exc)
                    continue
                total_boxes += len(rows)
            valid.append(pair)
        return valid, total_boxes

    def _materialise(self, assignment: dict[str, list[Pair]]) -> None:
        cfg = self.config
        for split, items in assignment.items():
            img_out = cfg.processed_dir / "images" / split
            lbl_out = cfg.processed_dir / "labels" / split
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            for pair in items:
                self._place(pair.image, img_out / pair.image.name)
                dst_label = lbl_out / f"{pair.image.stem}.txt"
                if pair.label.is_file():
                    self._place(pair.label, dst_label)
                else:
                    dst_label.write_text("", encoding="utf-8")  # background sample

    def _place(self, src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if self.config.copy_images:
            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.abspath(src), dst)

    def _write_dataset_yaml(self, assignment: dict[str, list[Pair]]) -> Path:
        cfg = self.config
        descriptor = {
            "path": str(cfg.processed_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": cfg.num_classes,
            "names": dict(enumerate(cfg.names)),
        }
        cfg.processed_dir.mkdir(parents=True, exist_ok=True)
        out = cfg.processed_dir / "dataset.yaml"
        with out.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(descriptor, fh, sort_keys=False)
        return out
