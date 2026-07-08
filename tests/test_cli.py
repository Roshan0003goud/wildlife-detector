"""Smoke tests for the CLI argument parser and lightweight commands."""

from __future__ import annotations

import pytest

from wildlife_detector.cli import build_parser, main


def test_parser_requires_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_prepare_defaults():
    args = build_parser().parse_args(["prepare"])
    assert args.config == "configs/data.yaml"
    assert callable(args.func)


def test_video_parses_tracking_flags():
    args = build_parser().parse_args(
        ["video", "--weights", "w.pt", "--source", "0", "--no-track", "--max-age", "10"]
    )
    assert args.no_track is True
    assert args.max_age == 10
    assert args.source == "0"


def test_info_command_runs():
    # `info` only inspects the environment; it must never raise.
    assert main(["info"]) == 0


def test_detect_requires_weights_and_source():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["detect"])
