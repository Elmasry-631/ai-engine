"""Gradio launch entrypoint.

Note: this file previously duplicated app inference code. It now reuses
`app.iface` to avoid divergence and keep prediction behavior consistent.
"""

from app import iface


if __name__ == "__main__":
    iface.launch()
