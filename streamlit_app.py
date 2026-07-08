"""Root entry point for hosted deployment.

Streamlit Community Cloud (and Hugging Face Spaces) auto-detect a top-level
``streamlit_app.py``, so this thin shim runs the real app in ``apps/`` — meaning
the deploy form needs no custom "main file path". The actual UI lives in
``apps/streamlit_app.py``.
"""

from __future__ import annotations

import runpy
from pathlib import Path

_APP = Path(__file__).resolve().parent / "apps" / "streamlit_app.py"
runpy.run_path(str(_APP), run_name="__main__")
