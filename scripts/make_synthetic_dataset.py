#!/usr/bin/env python
"""Generate a small *synthetic* YOLO dataset for smoke-testing the pipeline.

This is **not** training data — it draws coloured shapes on noise so you can
exercise `prepare` / `evaluate` / `detect` end-to-end without downloading a real
dataset. Each "animal" class is a distinct colour; labels are exact.

Usage:
    python scripts/make_synthetic_dataset.py --out data/raw --num 40
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Keep in sync with configs/data.yaml (first few classes are enough for a smoke test).
CLASS_COLORS = {
    0: (166, 206, 87),   # deer
    1: (120, 94, 66),    # wild_boar
    2: (150, 150, 160),  # elephant
    3: (90, 60, 40),     # bear
    4: (210, 180, 120),  # monkey
}


def _draw_animal(
    draw: ImageDraw.ImageDraw, w: int, h: int, cls: int
) -> tuple[float, float, float, float]:
    bw = random.randint(w // 8, w // 3)
    bh = random.randint(h // 8, h // 3)
    x1 = random.randint(0, w - bw)
    y1 = random.randint(0, h - bh)
    x2, y2 = x1 + bw, y1 + bh
    draw.ellipse([x1, y1, x2, y2], fill=CLASS_COLORS[cls], outline=(20, 20, 20), width=3)
    # Return normalised xywhn.
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    return cx, cy, bw / w, bh / h


def generate(out_dir: Path, num_images: int, size: int, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        noise = (np.random.rand(size, size, 3) * 60 + 40).astype("uint8")
        img = Image.fromarray(noise)
        draw = ImageDraw.Draw(img)

        rows = []
        for _ in range(random.randint(1, 3)):
            cls = random.choice(list(CLASS_COLORS))
            cx, cy, bw, bh = _draw_animal(draw, size, size, cls)
            rows.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        img.save(images_dir / f"synthetic_{i:04d}.jpg", quality=90)
        (labels_dir / f"synthetic_{i:04d}.txt").write_text("\n".join(rows), encoding="utf-8")

    print(f"Wrote {num_images} synthetic images + labels to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--num", type=int, default=40)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.out, args.num, args.size, args.seed)


if __name__ == "__main__":
    main()
