#!/usr/bin/env python3
"""Real CUDA Cellpose + Streamlit AppTest smoke for self-hosted GPU runners."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from streamlit.testing.v1 import AppTest


def _synthetic_png_bytes() -> bytes:
    image = np.zeros((160, 160), dtype=np.uint8)
    cv2.circle(image, (45, 55), 16, 200, -1)
    cv2.circle(image, (108, 95), 20, 215, -1)
    cv2.ellipse(image, (88, 38), (14, 9), 25, 0, 360, 185, -1)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("failed to encode synthetic PNG")
    return encoded.tobytes()


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this runner")

    app_path = Path(__file__).resolve().parents[1] / "app_streamlit.py"
    at = AppTest.from_file(app_path).run(timeout=180)
    if at.exception:
        raise RuntimeError(f"app startup exception: {at.exception}")

    at.sidebar.selectbox[0].select("cellpose")
    if at.sidebar.checkbox:
        at.sidebar.checkbox[0].set_value(True)
    at.file_uploader[0].upload("gpu-e2e.png", _synthetic_png_bytes(), "image/png")
    at.run(timeout=600)

    if at.exception:
        raise RuntimeError(f"Cellpose UI exception: {at.exception}")
    if at.error:
        raise RuntimeError(f"Cellpose UI errors: {[getattr(x, 'value', str(x)) for x in at.error]}")

    metrics = {getattr(metric, "label", ""): getattr(metric, "value", "") for metric in at.metric}
    if "Objects" not in metrics:
        raise RuntimeError(f"Objects metric missing; metrics={metrics}")

    downloads = [getattr(button, "label", "") for button in at.download_button]
    if "Download run summary JSON" not in downloads:
        raise RuntimeError(f"run-summary download missing; downloads={downloads}")

    print({
        "cuda_device": torch.cuda.get_device_name(0),
        "metrics": metrics,
        "downloads": downloads,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
