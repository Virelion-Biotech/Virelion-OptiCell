"""Acquisition-artifact metrics for microscopy images.

These metrics are descriptive QC signals, not automatic claims about biological
quality or instrument failure. They are intentionally modality-agnostic.
"""
from __future__ import annotations

import numpy as np


def _normalized_gradient(values: np.ndarray, axis: int) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or min(arr.shape) < 2:
        return 0.0
    profile = np.nanmean(arr, axis=axis)
    denominator = float(np.nanmean(np.abs(profile))) or 1.0
    return float(np.nanstd(np.diff(profile)) / denominator)


def acquisition_artifact_metrics(image: np.ndarray) -> dict[str, float]:
    """Return clipping, hot-pixel, background-gradient and illumination metrics."""
    arr = np.asarray(image)
    if arr.ndim > 2:
        arr = np.nanmean(arr[..., :3], axis=-1)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("image must be a non-empty 2-D array or HxWxC array")
    work = arr.astype(np.float32, copy=False)
    finite = np.isfinite(work)
    if not finite.any():
        raise ValueError("image contains no finite pixels")
    values = work[finite]
    lo, hi = float(np.min(values)), float(np.max(values))
    span = hi - lo
    if span <= 0:
        normalized = np.zeros_like(work)
    else:
        normalized = np.clip((work - lo) / span, 0.0, 1.0)
    return {
        "low_clip_fraction": float(np.mean(normalized <= 0.001)),
        "high_clip_fraction": float(np.mean(normalized >= 0.999)),
        "hot_pixel_fraction": float(np.mean(normalized >= 0.9995)),
        "horizontal_gradient_cv": _normalized_gradient(work, axis=0),
        "vertical_gradient_cv": _normalized_gradient(work, axis=1),
        "global_coefficient_of_variation": float(np.std(values) / (np.mean(np.abs(values)) or 1.0)),
        "background_percentile_1": float(np.percentile(values, 1.0)),
        "background_percentile_5": float(np.percentile(values, 5.0)),
    }


def artifact_burden_score(metrics: dict[str, float]) -> float:
    """Convert artifact metrics to a 0-100 descriptive quality score."""
    penalties = (
        min(25.0, metrics.get("low_clip_fraction", 0.0) * 50.0)
        + min(25.0, metrics.get("high_clip_fraction", 0.0) * 50.0)
        + min(15.0, metrics.get("hot_pixel_fraction", 0.0) * 300.0)
        + min(15.0, metrics.get("horizontal_gradient_cv", 0.0) * 100.0)
        + min(15.0, metrics.get("vertical_gradient_cv", 0.0) * 100.0)
    )
    return float(np.clip(100.0 - penalties, 0.0, 100.0))


__all__ = ["acquisition_artifact_metrics", "artifact_burden_score"]
