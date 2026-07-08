# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Wildlife detector runtime image.
# CPU inference by default; for GPU, swap the base for an nvidia/cuda image and
# install the matching torch build.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# OpenCV needs a couple of shared libraries at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WILDLIFE_LOG_LEVEL=INFO

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip && pip install .

# Bring in configs + helper apps/scripts.
COPY configs/ ./configs/
COPY apps/ ./apps/
COPY scripts/ ./scripts/

ENTRYPOINT ["wildlife-detect"]
CMD ["--help"]
