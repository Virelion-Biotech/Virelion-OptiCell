"""Texture and heterogeneity features for microscopy phenotyping."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _resolve_intensity_range(arr: np.ndarray, intensity_range: tuple[float, float] | None) -> tuple[float, float]:
    values = np.asarray(arr, dtype=np.float32)
    if intensity_range is not None:
        low, high = map(float, intensity_range)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("intensity_range must contain finite low < high")
        return low, high
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return float(info.min), float(info.max)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("image contains no finite pixels")
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return low, low + 1.0
    return low, high


def _histogram_entropy(values: np.ndarray, *, intensity_range: tuple[float, float]) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    low, high = intensity_range
    counts, _ = np.histogram(finite, bins=256, range=(low, high), density=False)
    probabilities = counts[counts > 0].astype(float)
    probabilities /= probabilities.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def basic_texture_features(gray: np.ndarray, intensity_range: tuple[float, float] | None = None) -> dict[str, float]:
    """Dependency-light texture descriptors for a 2-D grayscale image."""
    arr = np.asarray(gray, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("gray must be a non-empty 2-D image")
    if not np.isfinite(arr).all():
        raise ValueError("gray must contain only finite values")
    hist_range = _resolve_intensity_range(np.asarray(gray), intensity_range)
    values = arr.ravel()
    entropy = _histogram_entropy(values, intensity_range=hist_range)
    gx = np.diff(arr, axis=1)
    gy = np.diff(arr, axis=0)
    grad = np.concatenate([gx.ravel(), gy.ravel()])
    return {
        "intensity_entropy": entropy,
        "local_std_global": float(arr.std()),
        "gradient_mean_abs": float(np.mean(np.abs(grad))) if grad.size else 0.0,
        "gradient_std": float(grad.std()) if grad.size else 0.0,
        "edge_fraction_proxy": float(np.mean(np.abs(grad) > max(1.0, grad.std()))) if grad.size else 0.0,
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def object_texture_features(
    gray: np.ndarray,
    labels: np.ndarray,
    intensity_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Compute compact texture/heterogeneity features per segmented object."""
    image = np.asarray(gray, dtype=np.float32)
    masks = np.asarray(labels)
    if image.ndim != 2 or masks.shape != image.shape:
        raise ValueError("gray and labels must be matching 2-D arrays")
    if not np.isfinite(image).all():
        raise ValueError("gray must contain only finite values")
    hist_range = _resolve_intensity_range(np.asarray(gray), intensity_range)
    gy, gx = np.gradient(image)
    rows = []
    for label in np.unique(masks):
        if label <= 0:
            continue
        mask = masks == label
        values = image[mask]
        if values.size == 0:
            continue
        rows.append({
            "label": int(label),
            "texture_entropy": _histogram_entropy(values, intensity_range=hist_range),
            "texture_intensity_std": float(values.std()),
            "texture_p10": float(np.percentile(values, 10)),
            "texture_p90": float(np.percentile(values, 90)),
            "texture_gradient_mean": float(np.hypot(gx[mask], gy[mask]).mean()),
            "texture_gradient_std": float(np.hypot(gx[mask], gy[mask]).std()),
        })
    return pd.DataFrame(rows)
