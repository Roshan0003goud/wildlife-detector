"""Small logging helper so every entry-point gets consistent, readable output."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str = "wildlife_detector") -> logging.Logger:
    """Return a process-wide configured logger.

    The log level is controlled by the ``WILDLIFE_LOG_LEVEL`` environment
    variable (default ``INFO``) so verbosity can be raised without code changes.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.getenv("WILDLIFE_LOG_LEVEL", "INFO").upper()
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DATE_FORMAT))
        root = logging.getLogger("wildlife_detector")
        root.setLevel(getattr(logging, level, logging.INFO))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True

    if name == "wildlife_detector" or name.startswith("wildlife_detector."):
        return logging.getLogger(name)
    return logging.getLogger(f"wildlife_detector.{name}")
