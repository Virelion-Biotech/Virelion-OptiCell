"""Acquisition-artifact metrics for microscopy images.

These metrics are descriptive QC signals, not automatic claims about biological
quality or instrument failure. They are intentionally modality-agnostic.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _normalized_gradient(values: np.ndarray, axis: int) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or min(arr.shape) < 2:
        return 0.0
    profile = np.nanmean(arr, axis=axis)
    denominator = float(np.nanmean(np.abs(profile))) or 1.0
    return float(np.nanstd(np.diff(profile)) / denominator)


def _collapse_channels(image: np.ndarray) -> np.ndarray:
    """Reduce an HxWxC image to intensity without silently accepting extra axes."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[2] == 0:
            raise ValueError("image has zero channels")
        channels = arr[..., : min(3, arr.shape[2])]
        return np.nanmean(channels.astype(np.float32, copy=False), axis=2)
    raise ValueError("image must be a non-empty 2-D array or HxWxC array")


def _native_clip_fractions(arr: np.ndarray, values: np.ndarray, intensity_range: Optional[tuple[float, float]]) -> tuple[float, float]:
    """Measure clipping only against known detector/intensity bounds.

    Integer dtypes have well-defined representable limits. Floating-point data
    do not, so clipping is reported as unavailable (0) unless an explicit
    ``intensity_range`` is supplied.
    """
    if intensity_range is not None:
        low, high = map(float, intensity_range)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("intensity_range must contain finite low < high")
    elif np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        low, high = float(info.min), float(info.max)
    else:
        return 0.0, 0.0

    finite = np.isfinite(arr)
    total = int(arr.size)
    if not total or not finite.any():
        return 0.0, 0.0
    return float(np.count_nonzero(arr[finite] <= low) / total), float(np.count_nonzero(arr[finite] >= high) / total)


def _local_hot_pixel_fraction(work: np.ndarray) -> float:
    """Detect isolated bright pixels relative to their local neighborhood."""
    finite = np.isfinite(work)
    if work.size == 0 or not finite.any() or min(work.shape) < 3:
        return 0.0

    safe = np.where(finite, work, np.nanmedian(work[finite])).astype(np.float32, copy=False)
    local_median = cv2.medianBlur(safe, 3)
    residual = safe - local_median
    valid_residual = residual[finite]
    mad = float(np.median(np.abs(valid_residual - np.median(valid_residual))))
    scale = 1.4826 * mad
    threshold = max(5.0, 6.0 * scale)
    hot = finite & (residual > threshold)

    # A hot-pixel metric should not label an extended bright biological
    # structure: retain only pixels that are sufficiently isolated from their
    # immediate neighborhood.
    neighbor_max = cv2.dilate(local_median, np.ones((3, 3), np.uint8))
    isolated = hot & ((work - neighbor_max) > max(2.0, threshold * 0.25))
    return float(np.count_nonzero(isolated) / work.size)


def acquisition_artifact_metrics(
    image: np.ndarray,
    intensity_range: Optional[tuple[float, float]] = None,
) -> dict[str, float]:
    """Return descriptive clipping, hot-pixel, gradient and illumination metrics.

    ``intensity_range`` can be supplied for floating-point images when the
    detector/display acquisition range is known. Integer images use their
    dtype limits automatically. Hot pixels are identified as local bright
    outliers rather than simply the globally brightest pixels.
    """
    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("image must be a non-empty 2-D array or HxWxC array")
    collapsed = _collapse_channels(arr)
    work = collapsed.astype(np.float32, copy=False)
    finite = np.isfinite(work)
    if not finite.any():
        raise ValueError("image contains no finite pixels")
    values = work[finite]
    low_clip, high_clip = _native_clip_fractions(collapsed, values, intensity_range)
    return {
        "low_clip_fraction": low_clip,
        "high_clip_fraction": high_clip,
        "hot_pixel_fraction": _local_hot_pixel_fraction(work),
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
