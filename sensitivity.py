"""Segmentation parameter sensitivity analysis."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def threshold_sensitivity(
    image: np.ndarray,
    thresholds: Sequence[float],
    segmenter: Callable[[np.ndarray, float], np.ndarray],
) -> pd.DataFrame:
    """Measure segmentation count stability across threshold parameters.

    ``segmenter`` must return a labelled mask where 0 is background. The function
    intentionally evaluates only count and foreground-fraction stability; it does
    not imply that a stable count is a correct segmentation.
    """
    array = np.asarray(image)
    if array.ndim == 0 or array.size == 0:
        raise ValueError("image must be a non-empty array")
    values = [float(value) for value in thresholds]
    if not values:
        raise ValueError("thresholds must contain at least one value")
    if any(not np.isfinite(value) for value in values):
        raise ValueError("thresholds must be finite")
    if len(set(values)) != len(values):
        raise ValueError("thresholds must be unique")
    if not callable(segmenter):
        raise TypeError("segmenter must be callable")
    rows: list[dict[str, float | int]] = []
    for threshold in values:
        labels = np.asarray(segmenter(array, threshold))
        if labels.shape != array.shape:
            raise ValueError("segmenter output shape must match image shape")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("segmenter output must contain integer instance IDs")
        foreground = labels > 0
        count = int(np.unique(labels[foreground]).size) if foreground.any() else 0
        rows.append(
            {
                "threshold": threshold,
                "object_count": count,
                "foreground_fraction": float(foreground.mean()),
            }
        )
    result = pd.DataFrame(rows).sort_values("threshold", kind="stable").reset_index(drop=True)
    count_mean = float(result["object_count"].mean())
    count_sd = float(result["object_count"].std(ddof=1)) if len(result) > 1 else 0.0
    result["count_cv"] = count_sd / count_mean if count_mean else np.nan
    result["foreground_range"] = float(result["foreground_fraction"].max() - result["foreground_fraction"].min())
    return result


__all__ = ["threshold_sensitivity"]
