"""Optional offline augmentation stage.

Training-time augmentation (mosaic, HSV, flips) is handled by the YOLOv5 trainer;
this module provides *offline* augmentation for classes that are under-represented
in the raw data — a cheap way to rebalance a custom dataset before training.

The bounding-box transforms are pure NumPy (unit-tested); pixel operations use
OpenCV, imported lazily.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wildlife_detector.utils.logging import get_logger

logger = get_logger(__name__)


def hflip_boxes(boxes: NDArray) -> NDArray:
    """Horizontally flip normalised boxes by mirroring the x-centre.

    Accepts either ``xywhn`` rows (4 cols) or labelled ``[cls, x, y, w, h]`` rows
    (5 cols); the class-id column is left untouched.
    """
    b = np.asarray(boxes, dtype=np.float64)
    if b.size == 0:
        return b.reshape(-1, b.shape[-1] if b.ndim == 2 else 5)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    if b.shape[1] not in (4, 5):
        raise ValueError(f"Expected 4- or 5-column boxes; got shape {b.shape}.")
    x_col = 1 if b.shape[1] == 5 else 0  # skip the class id column when present
    out = b.copy()
    out[:, x_col] = 1.0 - b[:, x_col]
    return out


def _augment_image(image: NDArray, do_flip: bool, brightness: float, saturation: float) -> NDArray:
    import cv2

    img = image
    if do_flip:
        img = cv2.flip(img, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_split(
    images_dir: Path,
    labels_dir: Path,
    copies: int = 1,
    seed: int = 42,
) -> int:
    """Generate ``copies`` augmented variants for each image in a split.

    Returns the number of new (image, label) pairs written. Augmented files are
    suffixed ``_augN`` so a second run is idempotent per ``copies`` value.
    """
    import cv2

    rng = random.Random(seed)
    written = 0
    originals = [p for p in sorted(images_dir.glob("*")) if "_aug" not in p.stem]

    for img_path in originals:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Unreadable image skipped: %s", img_path)
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        boxes = _read_boxes(label_path)

        for k in range(copies):
            do_flip = rng.random() < 0.5
            brightness = rng.uniform(0.7, 1.3)
            saturation = rng.uniform(0.7, 1.3)
            aug_img = _augment_image(image, do_flip, brightness, saturation)
            aug_boxes = hflip_boxes(boxes) if (do_flip and len(boxes)) else boxes

            stem = f"{img_path.stem}_aug{k}"
            cv2.imwrite(str(images_dir / f"{stem}{img_path.suffix}"), aug_img)
            _write_boxes(labels_dir / f"{stem}.txt", aug_boxes)
            written += 1

    logger.info("Offline augmentation wrote %d new samples", written)
    return written


def _read_boxes(label_path: Path) -> NDArray:
    if not label_path.is_file():
        return np.zeros((0, 5), dtype=np.float64)
    rows = [
        [float(v) for v in line.split()]
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5) if rows else np.zeros((0, 5))


def _write_boxes(label_path: Path, boxes: NDArray) -> None:
    lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
    label_path.write_text("\n".join(lines), encoding="utf-8")
