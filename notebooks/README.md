# Notebooks

Exploratory notebooks live here (kept out of the package to keep the library
importable and testable).

Suggested notebooks:

- `01_dataset_eda.ipynb` — class balance, box-size distribution, sample grids.
- `02_training_colab.ipynb` — the Google Colab training run used to produce the
  reported metrics (mount the dataset, call `wildlife-detect train`, log to
  TensorBoard/W&B).
- `03_error_analysis.ipynb` — inspect false positives/negatives from
  `runs/eval/metrics_val.json` and visualise the hardest images.

> Tip: import the installed package directly inside a notebook —
> `from wildlife_detector.detection import WildlifeDetector` — rather than
> copy-pasting code, so notebooks stay thin.
