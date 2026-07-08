"""Real-time wild animal detection with YOLOv5 and OpenCV.

The package is organised into focused sub-modules:

* :mod:`wildlife_detector.config`     - typed, validated configuration objects.
* :mod:`wildlife_detector.data`       - preprocessing / dataset-preparation pipeline.
* :mod:`wildlife_detector.training`   - thin, reproducible wrapper over YOLOv5 training.
* :mod:`wildlife_detector.detection`  - inference, OpenCV video runner, object tracking.
* :mod:`wildlife_detector.evaluation` - mAP / IoU / accuracy metrics and reporting.
* :mod:`wildlife_detector.utils`      - logging, box geometry, visualisation helpers.

Heavy third-party dependencies (torch, ultralytics, OpenCV) are imported lazily
inside the functions that need them, so lightweight utilities and configuration
can be imported without a full deep-learning stack installed.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Roshan Goud"

__all__ = ["__version__", "__author__"]
