"""Tests for the dataset preparation pipeline (pure logic)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wildlife_detector.config import DataConfig, SplitRatios
from wildlife_detector.data.prepare import (
    DatasetPreparer,
    LabelError,
    find_pairs,
    split_indices,
    validate_label_text,
)


def test_split_indices_partition_is_exhaustive_and_disjoint():
    train, val, test = split_indices(100, (0.8, 0.1, 0.1), seed=42)
    assert len(train) == 80 and len(val) == 10 and len(test) == 10
    combined = sorted(train + val + test)
    assert combined == list(range(100))  # every index used exactly once


def test_split_indices_is_deterministic():
    a = split_indices(50, (0.7, 0.2, 0.1), seed=1)
    b = split_indices(50, (0.7, 0.2, 0.1), seed=1)
    assert a == b


def test_split_indices_different_seed_differs():
    a = split_indices(50, (0.7, 0.2, 0.1), seed=1)
    b = split_indices(50, (0.7, 0.2, 0.1), seed=2)
    assert a != b


def test_validate_label_text_valid():
    rows = validate_label_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05", num_classes=3)
    assert len(rows) == 2
    assert rows[0][0] == 0


def test_validate_label_text_empty_is_ok():
    assert validate_label_text("\n  \n", num_classes=3) == []


@pytest.mark.parametrize("bad", [
    "0 0.5 0.5 0.2",            # too few values
    "0 0.5 0.5 0.2 0.2 0.1",    # too many values
    "5 0.5 0.5 0.2 0.2",        # class out of range
    "0 1.5 0.5 0.2 0.2",        # coord > 1
    "0 0.5 0.5 0.0 0.2",        # zero width
    "x 0.5 0.5 0.2 0.2",        # non-numeric
])
def test_validate_label_text_rejects_bad_rows(bad):
    with pytest.raises(LabelError):
        validate_label_text(bad, num_classes=3)


def test_find_pairs_matches_labels(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"x")
    (tmp_path / "images" / "b.png").write_bytes(b"x")
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2")
    pairs = find_pairs(tmp_path)
    assert len(pairs) == 2
    names = {p.image.name for p in pairs}
    assert names == {"a.jpg", "b.png"}
    # 'b' has no label file -> treated as background
    b_pair = next(p for p in pairs if p.image.stem == "b")
    assert not b_pair.label.is_file()


def test_dataset_preparer_end_to_end(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir(parents=True)
    for i in range(10):
        (raw / "images" / f"img{i}.jpg").write_bytes(b"fake-jpeg")
        (raw / "labels" / f"img{i}.txt").write_text("0 0.5 0.5 0.3 0.3")

    cfg = DataConfig(
        raw_dir=raw,
        processed_dir=tmp_path / "processed",
        names=["deer", "fox"],
        split=SplitRatios(0.8, 0.1, 0.1),
        seed=42,
        copy_images=True,
    )
    yaml_path = DatasetPreparer(cfg).run()
    assert yaml_path.is_file()
    # Splits materialised with mirrored labels.
    train_imgs = list((cfg.processed_dir / "images" / "train").glob("*.jpg"))
    train_lbls = list((cfg.processed_dir / "labels" / "train").glob("*.txt"))
    assert len(train_imgs) == 8
    assert len(train_lbls) == 8
