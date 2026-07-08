"""Thin, reproducible wrapper around the ultralytics YOLOv5 trainer.

The goal is not to reimplement training but to make *our* training runs
declarative and repeatable: everything comes from :class:`TrainConfig`, the
resolved run directory is logged, and the best checkpoint is published to a
stable ``weights/best.pt`` path so downstream detection/evaluation code never
has to guess where the latest model lives.
"""

from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from typing import Any

from wildlife_detector.config import TrainConfig
from wildlife_detector.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Train a YOLOv5 model from a :class:`TrainConfig`."""

    def __init__(self, config: TrainConfig, weights_dir: Path | str = "weights") -> None:
        self.config = config
        self.weights_dir = Path(weights_dir)

    def train(self) -> dict[str, Any]:
        """Run training and return a summary dict.

        ``ultralytics`` / ``torch`` are imported here so the rest of the package
        stays importable on machines without a GPU stack.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The 'ultralytics' package is required to train. Install it with "
                "`pip install -e .` or `pip install ultralytics`."
            ) from exc

        cfg = self.config
        if not cfg.data.is_file():
            raise FileNotFoundError(
                f"Dataset descriptor {cfg.data} not found. Run "
                f"`wildlife-detect prepare` first."
            )

        logger.info("Loading base model: %s", cfg.model)
        model = YOLO(cfg.model)

        kwargs = cfg.to_ultralytics_kwargs()
        logger.info("Starting training for %d epochs (imgsz=%d, batch=%d)",
                    cfg.epochs, cfg.imgsz, cfg.batch)
        results = model.train(**kwargs)

        best = self._publish_best(model)
        summary = {
            "run_dir": str(getattr(results, "save_dir", cfg.project)),
            "best_weights": str(best) if best else None,
            "metrics": self._extract_metrics(results),
        }
        logger.info("Training complete. Best weights: %s", summary["best_weights"])
        return summary

    def _publish_best(self, model: Any) -> Path | None:
        """Copy the run's ``best.pt`` to a stable ``weights/best.pt``."""
        trainer = getattr(model, "trainer", None)
        best = getattr(trainer, "best", None)
        if not best or not Path(best).is_file():
            logger.warning("Could not locate best.pt to publish.")
            return None
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        dst = self.weights_dir / "best.pt"
        shutil.copy2(best, dst)
        logger.info("Published best checkpoint -> %s", dst)
        return dst

    @staticmethod
    def _extract_metrics(results: Any) -> dict[str, float]:
        """Best-effort extraction of summary metrics across ultralytics versions."""
        out: dict[str, float] = {}
        box = getattr(getattr(results, "box", None), "__dict__", {})
        results_dict = getattr(results, "results_dict", {}) or {}
        for key, source_key in (
            ("precision", "metrics/precision(B)"),
            ("recall", "metrics/recall(B)"),
            ("map50", "metrics/mAP50(B)"),
            ("map50_95", "metrics/mAP50-95(B)"),
        ):
            if source_key in results_dict:
                out[key] = float(results_dict[source_key])
        # Fallbacks for attribute-style access.
        fallbacks = (
            ("map50", "map50"),
            ("map50_95", "map"),
            ("precision", "mp"),
            ("recall", "mr"),
        )
        for key, attr in fallbacks:
            if key not in out and attr in box:
                with contextlib.suppress(TypeError, ValueError):
                    out[key] = float(box[attr])
        return out
