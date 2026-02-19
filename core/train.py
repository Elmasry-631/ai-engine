"""Gradio launch entrypoint."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import iface

"""Gradio launch entrypoint.

Note: this file previously duplicated app inference code. It now reuses
`app.iface` to avoid divergence and keep prediction behavior consistent.
"""

from app import iface


if __name__ == "__main__":
    iface.launch()
