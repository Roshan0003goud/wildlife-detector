.DEFAULT_GOAL := help
PY := python
PKG := src/wildlife_detector

.PHONY: help install install-dev prepare train evaluate detect webcam app test lint format typecheck clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies + package
	$(PY) -m pip install -e .

install-dev: ## Install package with dev + app extras and pre-commit hooks
	$(PY) -m pip install -e ".[dev,app]"
	pre-commit install

prepare: ## Run the preprocessing pipeline (split + validate + write data.yaml)
	$(PY) -m wildlife_detector.cli prepare --config configs/data.yaml

train: ## Train the YOLOv5 model using configs/train.yaml
	$(PY) -m wildlife_detector.cli train --config configs/train.yaml

evaluate: ## Evaluate a trained checkpoint on the validation split
	$(PY) -m wildlife_detector.cli evaluate --config configs/train.yaml --weights weights/best.pt

detect: ## Run detection on a single image (IMG=path/to/image.jpg)
	$(PY) -m wildlife_detector.cli detect --weights weights/best.pt --source $(IMG)

webcam: ## Run real-time detection on the default webcam (SOURCE=0)
	$(PY) -m wildlife_detector.cli video --weights weights/best.pt --source 0 --show

app: ## Launch the Streamlit demo UI
	streamlit run apps/streamlit_app.py

test: ## Run the test suite with coverage
	$(PY) -m pytest --cov=wildlife_detector --cov-report=term-missing

lint: ## Lint with ruff
	ruff check .

format: ## Auto-format + fix imports with ruff
	ruff format .
	ruff check --fix .

typecheck: ## Static type-check with mypy
	mypy

clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
