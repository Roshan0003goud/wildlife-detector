"""Real-time video detection loop built on OpenCV.

Reads frames from a webcam, video file or network stream; runs the detector on
each frame; optionally tracks objects across frames; draws boxes + a HUD; and
writes and/or displays the annotated stream. Returns run statistics (frame
count, average FPS, unique animals seen) so the pipeline can be scripted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wildlife_detector.detection.detector import WildlifeDetector, detections_to_arrays
from wildlife_detector.detection.tracker import IoUTracker
from wildlife_detector.utils.logging import get_logger
from wildlife_detector.utils.viz import color_palette, draw_detections, draw_hud

logger = get_logger(__name__)


@dataclass
class VideoStats:
    frames: int = 0
    elapsed_s: float = 0.0
    total_detections: int = 0
    unique_tracks: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)

    @property
    def avg_fps(self) -> float:
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "elapsed_s": round(self.elapsed_s, 2),
            "avg_fps": round(self.avg_fps, 2),
            "total_detections": self.total_detections,
            "unique_tracks": self.unique_tracks,
            "class_counts": self.class_counts,
        }


class VideoDetector:
    """Drive a :class:`WildlifeDetector` over a video source frame by frame."""

    def __init__(
        self,
        detector: WildlifeDetector,
        tracker: IoUTracker | None = None,
        draw: bool = True,
        show_hud: bool = True,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.draw = draw
        self.show_hud = show_hud

    @staticmethod
    def _open_source(source: str | int) -> Any:
        import cv2

        # A bare integer (or digit string) means a local camera index.
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source!r}")
        return cap

    def run(
        self,
        source: str | int,
        output: str | Path | None = None,
        show: bool = False,
        max_frames: int | None = None,
    ) -> VideoStats:
        import cv2

        cap = self._open_source(source)
        writer = None
        palette = color_palette(len(self.detector.class_names))
        seen_tracks: set[int] = set()
        stats = VideoStats()
        start = time.perf_counter()

        try:
            if output is not None:
                writer = self._make_writer(cap, output)

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                detections = self.detector.predict(frame)
                boxes, class_ids, scores = detections_to_arrays(detections)

                track_ids = None
                if self.tracker is not None:
                    track_ids = self.tracker.update(boxes, class_ids)
                    for tid, det in zip(track_ids, detections):
                        det.track_id = int(tid)
                        seen_tracks.add(int(tid))

                stats.frames += 1
                stats.total_detections += len(detections)
                for det in detections:
                    stats.class_counts[det.class_name] = (
                        stats.class_counts.get(det.class_name, 0) + 1
                    )

                if self.draw and len(detections):
                    draw_detections(
                        frame, boxes, class_ids, scores,
                        self.detector.class_names, track_ids, palette,
                    )
                if self.show_hud:
                    inst_fps = stats.frames / max(time.perf_counter() - start, 1e-6)
                    draw_hud(frame, inst_fps, len(detections))

                if writer is not None:
                    writer.write(frame)
                if show:
                    cv2.imshow("Wildlife Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Interrupted by user (q).")
                        break

                if max_frames is not None and stats.frames >= max_frames:
                    break
        finally:
            stats.elapsed_s = time.perf_counter() - start
            stats.unique_tracks = len(seen_tracks)
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        logger.info(
            "Processed %d frames in %.1fs (%.1f FPS); %d detections, %d unique animals",
            stats.frames, stats.elapsed_s, stats.avg_fps,
            stats.total_detections, stats.unique_tracks,
        )
        return stats

    def _make_writer(self, cap: Any, output: str | Path) -> Any:
        import cv2

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        logger.info("Writing annotated video to %s (%dx%d @ %.1f FPS)", output, width, height, fps)
        return cv2.VideoWriter(str(output), fourcc, fps, (width, height))
