"""Tests for configuration loading + validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from wildlife_detector.config import ConfigError, DataConfig, TrainConfig


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_data_config_roundtrip(tmp_path: Path):
    path = _write(tmp_path, "data.yaml", """
        raw_dir: data/raw
        processed_dir: data/processed
        split: {train: 0.8, val: 0.1, test: 0.1}
        seed: 7
        names: {0: deer, 1: fox}
    """)
    cfg = DataConfig.from_yaml(path)
    assert cfg.num_classes == 2
    assert cfg.names == ["deer", "fox"]
    assert cfg.seed == 7


def test_data_config_names_as_list(tmp_path: Path):
    path = _write(tmp_path, "data.yaml", """
        raw_dir: r
        processed_dir: p
        names: [deer, fox, bear]
    """)
    assert DataConfig.from_yaml(path).names == ["deer", "fox", "bear"]


def test_split_must_sum_to_one(tmp_path: Path):
    path = _write(tmp_path, "data.yaml", """
        raw_dir: r
        processed_dir: p
        split: {train: 0.9, val: 0.2, test: 0.1}
        names: [deer]
    """)
    with pytest.raises(ConfigError):
        DataConfig.from_yaml(path)


def test_non_contiguous_class_ids_raise(tmp_path: Path):
    path = _write(tmp_path, "data.yaml", """
        raw_dir: r
        processed_dir: p
        names: {0: deer, 2: fox}
    """)
    with pytest.raises(ConfigError):
        DataConfig.from_yaml(path)


def test_train_config_defaults_and_kwargs(tmp_path: Path):
    path = _write(tmp_path, "train.yaml", """
        data: data/processed/dataset.yaml
        epochs: 50
        batch: 8
        imgsz: 640
        augment: {mosaic: 1.0, hsv_h: 0.015}
    """)
    cfg = TrainConfig.from_yaml(path)
    assert cfg.epochs == 50
    kwargs = cfg.to_ultralytics_kwargs()
    assert kwargs["epochs"] == 50
    assert kwargs["mosaic"] == 1.0  # augment flattened in
    assert kwargs["data"] == "data/processed/dataset.yaml"


def test_imgsz_must_be_multiple_of_32(tmp_path: Path):
    path = _write(tmp_path, "train.yaml", """
        data: d.yaml
        imgsz: 100
    """)
    with pytest.raises(ConfigError):
        TrainConfig.from_yaml(path)


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        DataConfig.from_yaml("/nonexistent/data.yaml")
