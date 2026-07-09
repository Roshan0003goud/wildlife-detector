"""Streamlit demo UI for the wildlife detector.

Run with:  streamlit run apps/streamlit_app.py
Requires the optional 'app' extra:  pip install -e ".[app]"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the src-layout package importable when the app is launched from the repo
# root (e.g. on Streamlit Community Cloud) without a prior `pip install`.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from wildlife_detector.detection import WildlifeDetector  # noqa: E402
from wildlife_detector.utils.viz import color_palette, draw_detections  # noqa: E402

# A pretrained YOLO model that ultralytics auto-downloads, so the hosted demo
# works even before a custom model has been trained.
DEMO_MODEL = "yolov5su.pt"

st.set_page_config(page_title="Wildlife Detector", page_icon="🦌", layout="wide")


def _default_weights() -> str:
    """Prefer a trained custom checkpoint; otherwise the downloadable demo model."""
    return "weights/best.pt" if Path("weights/best.pt").is_file() else DEMO_MODEL


@st.cache_resource(show_spinner="Loading model…")
def load_detector(weights: str, conf: float, iou: float) -> WildlifeDetector:
    det = WildlifeDetector(weights, conf=conf, iou=iou)
    det.load()
    return det


def main() -> None:
    st.title("🦌 Wild Animal Detection — YOLOv5")
    st.caption("Upload an image and the model will locate and label the animals in it.")

    with st.sidebar:
        st.header("Configuration")
        weights = st.text_input("Weights path or model name", value=_default_weights())
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        iou = st.slider("NMS IoU threshold", 0.1, 0.9, 0.45, 0.05)
        st.markdown("---")
        st.markdown("Train a custom model with `wildlife-detect train`, then point "
                    "the field above at `weights/best.pt` for the 10 wildlife classes.")

    using_demo = not Path(weights).is_file()
    if using_demo:
        st.info(
            "ℹ️ **Demo mode** — running a pretrained YOLO model (auto-downloaded), so "
            "this detects general animal classes (bird, cat, dog, horse, sheep, cow, "
            "elephant, bear, zebra, giraffe). Train on the custom dataset and set the "
            "weights to `weights/best.pt` to detect the project's 10 wildlife species."
        )

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is None:
        st.info("Upload an image to run detection.")
        return

    from PIL import Image  # local import keeps startup light

    image = np.array(Image.open(uploaded).convert("RGB"))[:, :, ::-1]  # RGB -> BGR

    # Surface the *real* error in the UI (Streamlit redacts uncaught exceptions),
    # so deployment issues are diagnosable without digging through server logs.
    try:
        detector = load_detector(weights, conf, iou)
        start = time.perf_counter()
        detections = detector.predict(image.copy())
        elapsed_ms = (time.perf_counter() - start) * 1000
    except Exception:  # noqa: BLE001 - we deliberately show any failure to the user
        import traceback
        st.error("Model failed to load or run. Full error:")
        st.code(traceback.format_exc())
        st.stop()
        return

    annotated = draw_detections(
        image.copy(),
        [d.box for d in detections],
        [d.class_id for d in detections],
        [d.score for d in detections],
        detector.class_names,
        palette=color_palette(len(detector.class_names)),
    )

    col1, col2 = st.columns(2)
    col1.subheader("Input")
    col1.image(uploaded, use_container_width=True)
    col2.subheader(f"Detections ({len(detections)})")
    col2.image(annotated[:, :, ::-1], use_container_width=True)  # BGR -> RGB

    st.metric("Inference time", f"{elapsed_ms:.0f} ms")
    if detections:
        st.dataframe(
            {
                "class": [d.class_name for d in detections],
                "confidence": [round(d.score, 3) for d in detections],
                "box (xyxy)": [tuple(round(v, 1) for v in d.box) for d in detections],
            },
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
