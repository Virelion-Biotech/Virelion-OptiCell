#!/usr/bin/env python3
"""Real Cellpose smoke test intended for scheduled/manual CI, not mocked unit tests."""
from __future__ import annotations

import numpy as np

from qc_pipeline import CellposeSegmenter


def main() -> int:
    image = np.zeros((128, 128), dtype=np.uint8)
    yy, xx = np.ogrid[:128, :128]
    image[(xx - 42) ** 2 + (yy - 50) ** 2 <= 12 ** 2] = 180
    image[(xx - 88) ** 2 + (yy - 78) ** 2 <= 15 ** 2] = 210
    segmenter = CellposeSegmenter(model_type="cpsam", gpu=False)
    result = segmenter.segment(image)
    if result.labels.shape != image.shape:
        raise RuntimeError("Cellpose returned an unexpected mask shape")
    print({"objects": result.count, "quality_score": result.quality_score, "method": result.method})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
