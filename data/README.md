# Dataset

The model is trained on a **custom dataset of 2,000+ annotated wildlife images**
labelled in **YOLO format**. Raw data is intentionally excluded from version
control (see `.gitignore`); only the folder structure is tracked.

## Expected layout

Drop your annotated data under `data/raw/` like this:

```
data/raw/
├── images/
│   ├── img_0001.jpg
│   ├── img_0002.jpg
│   └── ...
└── labels/
    ├── img_0001.txt
    ├── img_0002.txt
    └── ...
```

Each `labels/<name>.txt` corresponds to `images/<name>.jpg` and contains one
row per bounding box:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are **normalised to `[0, 1]`** relative to image dimensions.
Class ids map to the `names` list in [`configs/data.yaml`](../configs/data.yaml).

Empty `.txt` files are allowed and denote background/negative images.

## Building the splits

```bash
wildlife-detect prepare --config configs/data.yaml
```

This validates every label, performs a reproducible 80/10/10 split, and writes
`data/processed/dataset.yaml` (the descriptor consumed by training).

## Where to get data

Public sources compatible with this format include
[iWildCam](https://github.com/visipedia/iwildcam_comp),
[LILA BC camera-trap datasets](https://lila.science/), and
[Roboflow Universe](https://universe.roboflow.com/) wildlife projects. Export as
"YOLOv5 PyTorch" to match the layout above.
