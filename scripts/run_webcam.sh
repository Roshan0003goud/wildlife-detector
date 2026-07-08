#!/usr/bin/env bash
# Convenience wrapper: real-time detection + tracking on the default webcam.
# Usage: ./scripts/run_webcam.sh [weights] [source]
set -euo pipefail

WEIGHTS="${1:-weights/best.pt}"
SOURCE="${2:-0}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Weights not found at '$WEIGHTS'. Train a model first (make train)." >&2
  exit 1
fi

exec wildlife-detect video \
  --weights "$WEIGHTS" \
  --source "$SOURCE" \
  --show \
  --conf 0.25
