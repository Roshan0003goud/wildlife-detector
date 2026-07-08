"""Typed, validated configuration objects loaded from YAML.

Using dataclasses (rather than passing raw dicts around) gives us three things:
IDE auto-completion, a single place to validate user input, and immutability of
the config once loaded. The module deliberately depends only on the standard
library + PyYAML so it can be imported anywhere without the ML stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing keys or has invalid values."""


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a top-level mapping.")
    return data


def _normalise_names(names: Any) -> list[str]:
    """Accept either a list of names or an ``{id: name}`` mapping and return a list.

    When a mapping is given, ids must be a contiguous ``0..N-1`` range so the
    resulting list index matches the class id used in label files.
    """
    if isinstance(names, list):
        if not names:
            raise ConfigError("`names` must contain at least one class.")
        return [str(n) for n in names]
    if isinstance(names, dict):
        try:
            items = sorted(((int(k), str(v)) for k, v in names.items()))
        except (TypeError, ValueError) as exc:  # non-integer keys
            raise ConfigError("`names` mapping keys must be integer class ids.") from exc
        expected = list(range(len(items)))
        if [k for k, _ in items] != expected:
            raise ConfigError(
                f"`names` ids must be a contiguous range 0..{len(items) - 1}; got "
                f"{[k for k, _ in items]}."
            )
        return [v for _, v in items]
    raise ConfigError("`names` must be a list or an {id: name} mapping.")


# --------------------------------------------------------------------------- #
# Dataset configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"Split ratios must sum to 1.0 (got {total:.3f}).")
        if min(self.train, self.val, self.test) < 0:
            raise ConfigError("Split ratios must be non-negative.")


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    processed_dir: Path
    names: list[str]
    split: SplitRatios = field(default_factory=SplitRatios)
    seed: int = 42
    copy_images: bool = False

    @property
    def num_classes(self) -> int:
        return len(self.names)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        raw = _load_yaml(path)
        try:
            split = SplitRatios(**(raw.get("split") or {}))
            return cls(
                raw_dir=Path(raw["raw_dir"]),
                processed_dir=Path(raw["processed_dir"]),
                names=_normalise_names(raw["names"]),
                split=split,
                seed=int(raw.get("seed", 42)),
                copy_images=bool(raw.get("copy_images", False)),
            )
        except KeyError as exc:
            raise ConfigError(f"Missing required key in data config: {exc}") from exc


# --------------------------------------------------------------------------- #
# Training / evaluation configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValConfig:
    conf: float = 0.001
    iou: float = 0.6
    report_conf: float = 0.25


@dataclass(frozen=True)
class TrainConfig:
    data: Path
    model: str = "yolov5su.pt"
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    patience: int = 25
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    cos_lr: bool = True
    augment: dict[str, float] = field(default_factory=dict)
    device: str = ""
    workers: int = 8
    project: str = "runs/train"
    name: str = "wildlife_yolov5"
    seed: int = 42
    exist_ok: bool = False
    val: ValConfig = field(default_factory=ValConfig)

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ConfigError("`epochs` must be a positive integer.")
        if self.batch <= 0:
            raise ConfigError("`batch` must be a positive integer.")
        if self.imgsz <= 0 or self.imgsz % 32 != 0:
            raise ConfigError("`imgsz` must be a positive multiple of 32.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        raw = _load_yaml(path)
        try:
            val = ValConfig(**(raw.get("val") or {}))
            return cls(
                data=Path(raw["data"]),
                model=str(raw.get("model", "yolov5su.pt")),
                epochs=int(raw.get("epochs", 100)),
                batch=int(raw.get("batch", 16)),
                imgsz=int(raw.get("imgsz", 640)),
                patience=int(raw.get("patience", 25)),
                optimizer=str(raw.get("optimizer", "auto")),
                lr0=float(raw.get("lr0", 0.01)),
                lrf=float(raw.get("lrf", 0.01)),
                momentum=float(raw.get("momentum", 0.937)),
                weight_decay=float(raw.get("weight_decay", 0.0005)),
                warmup_epochs=float(raw.get("warmup_epochs", 3.0)),
                cos_lr=bool(raw.get("cos_lr", True)),
                augment=dict(raw.get("augment") or {}),
                device=str(raw.get("device", "")),
                workers=int(raw.get("workers", 8)),
                project=str(raw.get("project", "runs/train")),
                name=str(raw.get("name", "wildlife_yolov5")),
                seed=int(raw.get("seed", 42)),
                exist_ok=bool(raw.get("exist_ok", False)),
                val=val,
            )
        except KeyError as exc:
            raise ConfigError(f"Missing required key in train config: {exc}") from exc

    def to_ultralytics_kwargs(self) -> dict[str, Any]:
        """Flatten this config into the keyword arguments ``YOLO.train`` expects."""
        kwargs: dict[str, Any] = {
            "data": str(self.data),
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "cos_lr": self.cos_lr,
            "device": self.device or None,
            "workers": self.workers,
            "project": self.project,
            "name": self.name,
            "seed": self.seed,
            "exist_ok": self.exist_ok,
        }
        # Augmentation keys are passed through verbatim (hsv_h, mosaic, ...).
        kwargs.update(self.augment)
        return kwargs
