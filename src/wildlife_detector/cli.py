"""Unified command-line interface for the wildlife detection pipeline.

Sub-commands:

    prepare    validate + split a raw dataset into a YOLO dataset.yaml
    augment    generate offline augmented copies of a split
    train      train a YOLOv5 model from configs/train.yaml
    evaluate   compute mAP / IoU / precision on a split
    detect     run detection on a single image or a folder of images
    video      run real-time detection + tracking on a video / webcam / stream
    info       print version and environment diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wildlife_detector import __version__
from wildlife_detector.utils.logging import get_logger

logger = get_logger("cli")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Command implementations
# --------------------------------------------------------------------------- #
def _cmd_prepare(args: argparse.Namespace) -> int:
    from wildlife_detector.config import DataConfig
    from wildlife_detector.data import DatasetPreparer

    config = DataConfig.from_yaml(args.config)
    yaml_path = DatasetPreparer(config).run()
    logger.info("Prepared dataset descriptor: %s", yaml_path)
    return 0


def _cmd_augment(args: argparse.Namespace) -> int:
    from wildlife_detector.config import DataConfig
    from wildlife_detector.data.augment import augment_split

    config = DataConfig.from_yaml(args.config)
    images = config.processed_dir / "images" / args.split
    labels = config.processed_dir / "labels" / args.split
    if not images.is_dir():
        logger.error("Split not found: %s (run `prepare` first)", images)
        return 1
    written = augment_split(images, labels, copies=args.copies, seed=config.seed)
    logger.info("Augmentation complete: %d new samples in the %s split", written, args.split)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from wildlife_detector.config import TrainConfig
    from wildlife_detector.training import Trainer

    config = TrainConfig.from_yaml(args.config)
    summary = Trainer(config, weights_dir=args.weights_dir).train()
    logger.info("Training summary:\n%s", json.dumps(summary, indent=2))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from wildlife_detector.config import TrainConfig
    from wildlife_detector.evaluation import Evaluator

    config = TrainConfig.from_yaml(args.config)
    evaluator = Evaluator(
        weights=args.weights,
        dataset_yaml=config.data,
        conf=config.val.conf,
        iou=config.val.iou,
        report_conf=config.val.report_conf,
        imgsz=config.imgsz,
        device=config.device or None,
    )
    evaluator.run(split=args.split, output_dir=args.output)
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    import cv2

    from wildlife_detector.detection import WildlifeDetector
    from wildlife_detector.utils.viz import color_palette, draw_detections

    detector = WildlifeDetector(
        args.weights, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        device=args.device or None,
    )
    sources = _collect_images(Path(args.source))
    if not sources:
        logger.error("No images found at %s", args.source)
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = color_palette(len(detector.class_names))
    total = 0

    for path in sources:
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Unreadable image skipped: %s", path)
            continue
        detections = detector.predict(frame)
        total += len(detections)
        logger.info("%s -> %d detection(s)", path.name, len(detections))
        for det in detections:
            logger.info("    %-12s %.2f  box=%s",
                        det.class_name, det.score, tuple(round(v, 1) for v in det.box))
        boxes = [d.box for d in detections]
        draw_detections(
            frame, boxes, [d.class_id for d in detections],
            [d.score for d in detections], detector.class_names, palette=palette,
        )
        out_path = out_dir / f"{path.stem}_pred{path.suffix}"
        cv2.imwrite(str(out_path), frame)
        if args.show:
            cv2.imshow("detection", frame)
            cv2.waitKey(0)
    if args.show:
        cv2.destroyAllWindows()
    logger.info("Done: %d image(s), %d detection(s). Annotated output in %s",
                len(sources), total, out_dir)
    return 0


def _cmd_video(args: argparse.Namespace) -> int:
    from wildlife_detector.detection import IoUTracker, VideoDetector, WildlifeDetector

    detector = WildlifeDetector(
        args.weights, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        device=args.device or None,
    )
    tracker = None if args.no_track else IoUTracker(
        iou_threshold=args.track_iou, max_age=args.max_age
    )
    runner = VideoDetector(detector, tracker=tracker)
    stats = runner.run(
        source=args.source, output=args.output, show=args.show, max_frames=args.max_frames
    )
    logger.info("Video stats:\n%s", json.dumps(stats.as_dict(), indent=2))
    return 0


def _cmd_info(_args: argparse.Namespace) -> int:
    logger.info("wildlife-detector v%s", __version__)
    logger.info("python %s", sys.version.split()[0])
    for mod in ("numpy", "yaml", "torch", "ultralytics", "cv2"):
        try:
            imported = __import__(mod)
            version = getattr(imported, "__version__", "unknown")
            logger.info("  %-12s %s", mod, version)
        except ImportError:
            logger.info("  %-12s not installed", mod)
    return 0


def _collect_images(source: Path) -> list[Path]:
    if source.is_dir():
        return [p for p in sorted(source.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source]
    return []


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wildlife-detect",
        description="Real-time wild animal detection with YOLOv5 + OpenCV.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Validate + split a raw dataset")
    p_prep.add_argument("--config", default="configs/data.yaml")
    p_prep.set_defaults(func=_cmd_prepare)

    p_aug = sub.add_parser("augment", help="Offline-augment a split")
    p_aug.add_argument("--config", default="configs/data.yaml")
    p_aug.add_argument("--split", default="train", choices=["train", "val", "test"])
    p_aug.add_argument("--copies", type=int, default=1)
    p_aug.set_defaults(func=_cmd_augment)

    p_train = sub.add_parser("train", help="Train a YOLOv5 model")
    p_train.add_argument("--config", default="configs/train.yaml")
    p_train.add_argument("--weights-dir", default="weights")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a checkpoint on a split")
    p_eval.add_argument("--config", default="configs/train.yaml")
    p_eval.add_argument("--weights", required=True)
    p_eval.add_argument("--split", default="val", choices=["train", "val", "test"])
    p_eval.add_argument("--output", default="runs/eval")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_det = sub.add_parser("detect", help="Detect on image(s)")
    p_det.add_argument("--weights", required=True)
    p_det.add_argument("--source", required=True, help="Image file or folder")
    p_det.add_argument("--output", default="outputs")
    _add_inference_args(p_det)
    p_det.add_argument("--show", action="store_true")
    p_det.set_defaults(func=_cmd_detect)

    p_vid = sub.add_parser("video", help="Real-time detection + tracking")
    p_vid.add_argument("--weights", required=True)
    p_vid.add_argument("--source", required=True, help="Webcam index, video path or stream URL")
    p_vid.add_argument("--output", default=None, help="Optional annotated .mp4 output")
    p_vid.add_argument("--show", action="store_true")
    p_vid.add_argument("--max-frames", type=int, default=None)
    p_vid.add_argument("--no-track", action="store_true", help="Disable object tracking")
    p_vid.add_argument("--track-iou", type=float, default=0.3)
    p_vid.add_argument("--max-age", type=int, default=30)
    _add_inference_args(p_vid)
    p_vid.set_defaults(func=_cmd_video)

    p_info = sub.add_parser("info", help="Print version + environment")
    p_info.set_defaults(func=_cmd_info)

    return parser


def _add_inference_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help='"", "cpu", "0", ...')


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        logger.warning("Interrupted.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
