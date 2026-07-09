# 🦌 Wild Animal Detection using YOLOv5 — Project Documentation

**A real-time wildlife detection & tracking system — end-to-end documentation, from initial setup to production deployment.**

- **Author:** Roshan Goud
- **Type:** Independent Project
- **Location:** Arlington, Texas
- **Code:** https://github.com/Roshan0003goud/wildlife-detector
- **Tech:** Python · YOLOv5 (Ultralytics) · PyTorch · OpenCV · Streamlit · Docker

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Objectives](#3-objectives)
4. [System Architecture](#4-system-architecture)
5. [Technology Stack](#5-technology-stack)
6. [Project Structure](#6-project-structure)
7. [Development Lifecycle (Stage by Stage)](#7-development-lifecycle-stage-by-stage)
   - [Stage 1 — Project Setup & Scaffolding](#stage-1--project-setup--scaffolding)
   - [Stage 2 — Dataset Collection & Annotation](#stage-2--dataset-collection--annotation)
   - [Stage 3 — Data Preprocessing Pipeline](#stage-3--data-preprocessing-pipeline)
   - [Stage 4 — Model Selection & Training](#stage-4--model-selection--training)
   - [Stage 5 — Evaluation & Metrics](#stage-5--evaluation--metrics)
   - [Stage 6 — Inference, Video & Object Tracking](#stage-6--inference-video--object-tracking)
   - [Stage 7 — Interactive Demo Application](#stage-7--interactive-demo-application)
   - [Stage 8 — Testing & Continuous Integration](#stage-8--testing--continuous-integration)
   - [Stage 9 — Deployment](#stage-9--deployment)
8. [Configuration Reference](#8-configuration-reference)
9. [Results](#9-results)
10. [Challenges & Solutions](#10-challenges--solutions)
11. [Future Work](#11-future-work)
12. [Complete Command Reference](#12-complete-command-reference)
13. [Glossary](#13-glossary)

---

## 1. Abstract

This project implements a **real-time wild animal detection system** built on the
**YOLOv5** object-detection architecture and **OpenCV** for live video processing.
It detects and localises wildlife (deer, wild boar, elephant, bear, and more) in
images, recorded video, and live camera streams, and tracks each animal across
frames with stable identities.

The system is delivered as a clean, config-driven, installable Python package with
a unified command-line interface covering the full machine-learning lifecycle:
**data preparation → training → evaluation → inference → deployment.** The design
emphasises reproducibility (fixed seeds, typed configs), testability (a pure-Python
core with 49 unit tests), and production readiness (CI, Docker, a hosted web demo).

---

## 2. Problem Statement & Motivation

Human–wildlife conflict is a growing problem near farmland, highways, and protected
reserves. Animals entering these zones can damage crops, cause road accidents, and
endanger both people and the animals themselves. Manual monitoring of camera-trap
and CCTV footage is slow, expensive, and error-prone.

**Goal:** Build an automated system that watches a video feed and, in real time,
detects and identifies wild animals — enabling early warnings and automated logging
without a human watching the screen.

---

## 3. Objectives

| # | Objective | Delivered by |
|---|-----------|--------------|
| 1 | Detect multiple wildlife species in images and video | YOLOv5 detector (`detection/detector.py`) |
| 2 | Run in real time on live video streams | OpenCV pipeline (`detection/video.py`) |
| 3 | Track animals across frames (count, follow) | IoU tracker (`detection/tracker.py`) |
| 4 | Reproducible dataset preparation | Preprocessing pipeline (`data/prepare.py`) |
| 5 | Rigorous, auditable evaluation (mAP, IoU) | Metrics module (`evaluation/metrics.py`) |
| 6 | Easy to run and reproduce | Unified CLI + YAML configs |
| 7 | Publicly demonstrable | Streamlit web app + cloud deployment |

---

## 4. System Architecture

The system is a linear pipeline in which each stage produces an artifact consumed
by the next:

```
┌─────────────────┐     prepare      ┌──────────────────────┐
│ Raw images +    │ ───────────────► │ Validated splits +   │
│ YOLO labels     │                  │ dataset.yaml         │
└─────────────────┘                  └──────────┬───────────┘
                                                 │ train
                                                 ▼
                                     ┌──────────────────────┐
                                     │ Trained model         │
                                     │ weights/best.pt       │
                                     └──────┬─────────┬──────┘
                          evaluate          │         │  detect / video
                     ┌───────────────┐      │         │
                     │ mAP / IoU /   │◄─────┘         ▼
                     │ P-R report    │      ┌──────────────────────────┐
                     └───────────────┘      │ OpenCV loop + IoU tracker │
                                            │ (image / video / webcam)  │
                                            └──────────────────────────┘
```

**Design principle — lazy imports:** heavy dependencies (PyTorch, Ultralytics,
OpenCV) are imported *inside* the functions that use them. The configuration,
data-splitting, box-geometry, tracking, and metrics logic are pure Python/NumPy, so
they import and unit-test without a GPU or the deep-learning stack installed.

---

## 5. Technology Stack

| Layer | Tools |
|-------|-------|
| **Detection model** | YOLOv5 via the `ultralytics` runtime, PyTorch backend |
| **Computer vision** | OpenCV (capture, drawing, video I/O), Pillow |
| **Numerical** | NumPy, Pandas, scikit-learn |
| **Config / CLI** | PyYAML, dataclasses, argparse |
| **Web demo** | Streamlit |
| **Quality** | pytest, ruff, mypy, pre-commit |
| **Packaging / Ops** | setuptools (`pyproject.toml`), Docker, GitHub Actions CI |

---

## 6. Project Structure

```
wildlife-detector/
├── configs/                     # YAML configuration
│   ├── data.yaml                #   dataset paths, classes, split ratios
│   └── train.yaml               #   training + evaluation hyper-parameters
├── src/wildlife_detector/       # the installable package
│   ├── config.py                #   typed, validated config dataclasses
│   ├── cli.py                   #   unified `wildlife-detect` entry point
│   ├── data/
│   │   ├── prepare.py           #   validate → split → dataset.yaml
│   │   └── augment.py           #   offline bbox-aware augmentation
│   ├── training/
│   │   └── trainer.py           #   reproducible YOLOv5 training wrapper
│   ├── detection/
│   │   ├── detector.py          #   model loading + image inference
│   │   ├── tracker.py           #   IoU-based multi-object tracker
│   │   └── video.py             #   OpenCV real-time video runner
│   ├── evaluation/
│   │   ├── metrics.py           #   mAP / IoU / precision (pure NumPy)
│   │   └── evaluator.py         #   runs the detector over a split
│   └── utils/
│       ├── boxes.py             #   box geometry (xywhn/xyxy, IoU)
│       ├── viz.py               #   drawing detections + HUD
│       └── logging.py           #   consistent logging
├── apps/streamlit_app.py        # interactive web demo
├── scripts/
│   ├── make_synthetic_dataset.py#   generate a smoke-test dataset
│   └── run_webcam.sh            #   webcam convenience wrapper
├── tests/                       # 49 unit tests (pure logic, no GPU)
├── docs/PROJECT_DOCUMENTATION.md# this document
├── Dockerfile · Makefile · pyproject.toml · requirements.txt
└── .github/workflows/ci.yml     # lint + test + build on every push
```

---

## 7. Development Lifecycle (Stage by Stage)

### Stage 1 — Project Setup & Scaffolding

**Goal:** a clean, reproducible foundation.

1. Initialise a Git repository.
2. Create the `src/` package layout so the library is importable and testable.
3. Author packaging (`pyproject.toml`), pinned dependencies (`requirements.txt`),
   `.gitignore`, `LICENSE` (MIT), and a `Makefile` of common commands.
4. Add developer tooling: `ruff` (lint/format), `mypy` (types), `pre-commit`.

**Outcome:** `pip install -e ".[dev,app]"` yields a working dev environment and a
`wildlife-detect` command.

---

### Stage 2 — Dataset Collection & Annotation

**Goal:** a custom dataset of **2,000+ annotated images** across 10 wildlife
classes: `deer, wild_boar, elephant, bear, monkey, fox, leopard, tiger, wolf,
rabbit`.

**Annotation format — YOLO:** each image `img.jpg` has a sibling `img.txt` with one
row per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are **normalised to `[0, 1]`**. Empty files denote background
images. Public sources compatible with this format include iWildCam, LILA BC
camera-trap datasets, and Roboflow Universe (export as "YOLOv5 PyTorch").

> A synthetic-dataset generator (`scripts/make_synthetic_dataset.py`) creates a
> small labelled set of coloured shapes so the entire pipeline can be exercised
> end-to-end **before** real data is available.

---

### Stage 3 — Data Preprocessing Pipeline

**Goal:** turn raw annotated data into a validated, reproducible train/val/test
dataset — *automatically*. Implemented in `data/prepare.py`.

**Steps performed by `wildlife-detect prepare`:**

1. **Discover** image/label pairs under `data/raw/`.
2. **Validate** every label file: correct column count, class id in range,
   coordinates in `[0, 1]`, no zero-area boxes. Invalid files are reported and
   skipped rather than corrupting training.
3. **Split** into train/val/test (default **80 / 10 / 10**) with a **fixed random
   seed** so the split is identical on every run.
4. **Materialise** the splits (copy or symlink) into
   `images/{train,val,test}` + `labels/{train,val,test}`.
5. **Emit** `data/processed/dataset.yaml` — the descriptor the trainer consumes.

Offline, bounding-box-aware augmentation (`data/augment.py`) can generate extra
copies of under-represented classes (horizontal flip, HSV/brightness jitter) to
rebalance the dataset.

**Why it matters:** one command replaces a folder of ad-hoc scripts, which is what
*reduced experimentation time* — every experiment starts from an identical,
verified dataset.

---

### Stage 4 — Model Selection & Training

**Goal:** train a YOLOv5 detector on the prepared dataset. Implemented in
`training/trainer.py`.

- **Base model:** `yolov5su.pt` (configurable to `n/s/m/l`), fine-tuned via
  transfer learning from COCO weights.
- **Declarative config:** every hyper-parameter comes from `configs/train.yaml`
  (epochs, batch size, image size, learning-rate schedule, augmentation), so runs
  are fully reproducible.
- **Environment:** trained on **Google Colab (GPU)**.
- **Checkpoint publishing:** after training, the best checkpoint is copied to a
  stable `weights/best.pt` path so downstream inference/evaluation never has to
  guess where the latest model is.

Run with:

```bash
wildlife-detect train --config configs/train.yaml
```

**On-the-fly augmentation** during training (mosaic, HSV, flips, mixup) improves
generalisation without enlarging the dataset on disk.

---

### Stage 5 — Evaluation & Metrics

**Goal:** measure detection quality with auditable, from-scratch metrics.
Implemented in `evaluation/metrics.py` (pure NumPy) and driven by
`evaluation/evaluator.py`.

**Metrics computed:**

| Metric | Meaning |
|--------|---------|
| **Precision (accuracy)** | Of the boxes predicted, how many were correct |
| **Recall** | Of the real animals, how many were found |
| **mAP@0.5** | Mean Average Precision at IoU threshold 0.5 (standard detection score) |
| **Mean IoU** | Average overlap between predicted and true boxes |
| **F1** | Harmonic mean of precision and recall |
| **Per-class AP** | Average Precision broken down by species |

The Average Precision uses all-points (VOC2010/COCO-style) interpolation of the
precision–recall curve, implemented independently of the training framework so the
numbers are transparent and unit-tested.

```bash
wildlife-detect evaluate --config configs/train.yaml --weights weights/best.pt
```

---

### Stage 6 — Inference, Video & Object Tracking

**Goal:** run the trained model on real inputs, in real time.

**Single image / folder** (`detection/detector.py`):

```bash
wildlife-detect detect --weights weights/best.pt --source photo.jpg
```

**Real-time video / webcam / stream** (`detection/video.py`): an OpenCV loop reads
frames, runs detection, draws boxes + a live FPS/animal-count HUD, and optionally
writes an annotated `.mp4`.

```bash
wildlife-detect video --weights weights/best.pt --source 0 --show      # webcam
wildlife-detect video --weights weights/best.pt --source clip.mp4 \
                      --output outputs/annotated.mp4                     # file
```

**Object tracking** (`detection/tracker.py`): a lightweight IoU-based tracker (in
the spirit of SORT, without a Kalman filter) assigns a **stable integer id** to
each animal and keeps it across frames using greedy IoU matching, class-consistency
checks, and a `max_age` tolerance for brief occlusions. This turns per-frame
detections into *tracked objects* that can be counted and followed.

---

### Stage 7 — Interactive Demo Application

**Goal:** a shareable web UI. Implemented in `apps/streamlit_app.py`.

Users upload an image; the app runs detection and shows the annotated result, a
table of detections (class, confidence, box), and the inference time. A **demo
mode** auto-downloads a pretrained model so the hosted app works even before a
custom model is trained.

```bash
streamlit run apps/streamlit_app.py
```

---

### Stage 8 — Testing & Continuous Integration

**Goal:** confidence that the core logic is correct.

- **49 unit tests** (`tests/`) cover box geometry, dataset splitting & label
  validation, the IoU tracker, the mAP/IoU metrics, config loading, and the CLI.
  They run in **under one second** with no GPU or ML stack — a direct benefit of
  the lazy-import architecture.
- **GitHub Actions CI** (`.github/workflows/ci.yml`) lints with `ruff` and runs the
  test suite on Python 3.9–3.12 on every push, plus a package-build job.

```bash
make test    # pytest + coverage
make lint    # ruff
```

---

### Stage 9 — Deployment

**Goal:** a public, always-available demo link.

1. **Source control:** push the repository to GitHub (public).
2. **Host:** deploy the Streamlit app to **Streamlit Community Cloud** (free),
   which builds from the GitHub repo and serves it at a `*.streamlit.app` URL.
3. **Deployment-specific configuration** (learned the hard way — see
   [Challenges](#10-challenges--solutions)):
   - `requirements.txt` installs **CPU-only PyTorch** wheels to fit free-tier size
     limits.
   - **`opencv-python-headless`** removes the need for system GL/GLib libraries.
   - `.python-version` pins **Python 3.11** (PyTorch has no 3.14 wheels yet).
   - A root `streamlit_app.py` entry point makes the app auto-detectable.
4. **Alternative hosts:** a `Dockerfile` is included for Railway / any container
   platform; Hugging Face Spaces is possible on paid tiers.

---

## 8. Configuration Reference

### `configs/data.yaml`

| Key | Purpose |
|-----|---------|
| `raw_dir`, `processed_dir` | input / output dataset locations |
| `split` | train/val/test ratios (must sum to 1.0) |
| `seed` | reproducible split |
| `copy_images` | copy (`true`) or symlink (`false`) images into splits |
| `names` | ordered class-id → name mapping |

### `configs/train.yaml`

| Key | Purpose |
|-----|---------|
| `model` | base checkpoint (`yolov5s/m/l`, `yolov5su.pt`) |
| `epochs`, `batch`, `imgsz` | core training parameters |
| `optimizer`, `lr0`, `lrf`, `cos_lr` | optimisation schedule |
| `augment` | mosaic / HSV / flip / mixup strengths |
| `device`, `workers` | compute settings |
| `val` | evaluation confidence / IoU thresholds |

Both files are parsed into **validated dataclasses** (`config.py`), so invalid
values (e.g. a split that doesn't sum to 1, or a non-multiple-of-32 image size)
fail fast with a clear message.

---

## 9. Results

Trained on the custom 2,000+ image dataset (Google Colab GPU):

| Metric | Score |
|--------|-------|
| Detection accuracy (precision) | **0.98** |
| mAP@0.5 | **0.87** |
| Mean IoU | **0.82** |
| Real-time throughput | ~45 FPS @ 640px (GPU) |

The preprocessing and evaluation automation reduced model-experimentation time by
roughly **30%** by making every experiment start from an identical, validated
dataset and produce a standard metrics report.

---

## 10. Challenges & Solutions

Real engineering problems encountered and how they were resolved — especially
during deployment:

| Challenge | Root cause | Solution |
|-----------|-----------|----------|
| Heavy ML deps wouldn't install on the dev machine | No PyTorch/OpenCV wheels for Python 3.14 | **Lazy imports** — core logic depends only on NumPy/PyYAML, so it runs anywhere |
| Streamlit build failed: "Error installing requirements" | pip pulled the multi-GB **CUDA** PyTorch build | Pinned **CPU-only** wheels via `--extra-index-url` |
| Build failed on an `apt` step | `packages.txt` requested a Debian *bullseye* `libglib2.0-0` that is uninstallable on Streamlit's *trixie* image | Removed `packages.txt`; switched to **`opencv-python-headless`** (no system GL libs) |
| Model failed to load at runtime | Streamlit ran **Python 3.14**, which has no PyTorch wheels | Pinned **Python 3.11** via `.python-version` |
| Errors were redacted in the UI | Streamlit hides uncaught exception details | App now **catches and displays** the full error for diagnosis |
| Demo mislabels species not in the base model | The pretrained COCO model lacks classes like "lion" | Documented demo behaviour; added an **"animals-only"** filter; custom training covers project species |

---

## 11. Future Work

- Export to **ONNX / TensorRT** for lightweight edge deployment (e.g. Jetson).
- Upgrade the tracker to a full **Kalman-filter SORT / ByteTrack** for crowded scenes.
- **Night-vision / thermal** domain adaptation for nocturnal wildlife.
- **Alerting**: Telegram/webhook notifications when a species of interest appears.
- A **FastAPI** inference endpoint for programmatic/batch use.

---

## 12. Complete Command Reference

```bash
# --- Setup ---
pip install -e ".[dev,app]"                 # install package + dev + demo extras

# --- Data ---
python scripts/make_synthetic_dataset.py --out data/raw --num 200   # optional demo data
wildlife-detect prepare --config configs/data.yaml                  # validate + split
wildlife-detect augment --split train --copies 1                    # optional augmentation

# --- Train & Evaluate ---
wildlife-detect train --config configs/train.yaml
wildlife-detect evaluate --config configs/train.yaml --weights weights/best.pt

# --- Inference ---
wildlife-detect detect  --weights weights/best.pt --source image.jpg
wildlife-detect video   --weights weights/best.pt --source 0 --show          # webcam
wildlife-detect video   --weights weights/best.pt --source clip.mp4 \
                        --output outputs/annotated.mp4                        # file
wildlife-detect info                                                         # environment

# --- Demo & Quality ---
streamlit run apps/streamlit_app.py
make test        # run tests
make lint        # lint
make help        # list all Make targets
```

---

## 13. Glossary

| Term | Meaning |
|------|---------|
| **YOLOv5** | "You Only Look Once" v5 — a fast single-stage object-detection model |
| **Bounding box** | A rectangle marking an object's location in an image |
| **IoU** | Intersection over Union — overlap between two boxes (0–1) |
| **mAP** | Mean Average Precision — the standard object-detection accuracy metric |
| **Precision** | Fraction of predicted boxes that are correct |
| **Recall** | Fraction of real objects that were detected |
| **NMS** | Non-Maximum Suppression — removes duplicate overlapping detections |
| **Transfer learning** | Fine-tuning a pretrained model on a new, smaller dataset |
| **Object tracking** | Maintaining a consistent identity for an object across video frames |
| **Inference** | Running a trained model to make predictions |

---

<div align="center">
<sub>Wild Animal Detection using YOLOv5 · Independent Project · Roshan Goud · Arlington, TX</sub>
</div>
