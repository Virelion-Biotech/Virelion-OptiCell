"""Higher-level quantitative microscopy analysis utilities.

These functions operate on OptiCell segmentation outputs and raw NumPy image
arrays. They intentionally remain GUI-free and dependency-light.
"""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

import cv2
import numpy as np
import pandas as pd


def nearest_neighbor_distances(features: pd.DataFrame) -> np.ndarray:
    """Return each object's nearest-neighbour centroid distance in pixels."""
    required = {"centroid_x", "centroid_y"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"features missing required columns: {sorted(missing)}")
    points = features[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    if not np.isfinite(points).all():
        raise ValueError("centroid coordinates must be finite")
    n = len(points)
    if n < 2:
        return np.full(n, np.nan, dtype=float)
    nearest = np.full(n, np.inf, dtype=float)
    for start in range(0, n, 1024):
        stop = min(start + 1024, n)
        block = points[start:stop]
        d2 = ((block[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        local_indices = np.arange(start, stop)
        d2[np.arange(stop - start), local_indices] = np.inf
        nearest[start:stop] = np.sqrt(d2.min(axis=1))
    return nearest


def add_spatial_features(features: pd.DataFrame, image_shape: Sequence[int]) -> pd.DataFrame:
    """Add nearest-neighbour, normalized coordinates and density measures."""
    shape = tuple(image_shape)
    if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("image_shape must contain positive height and width")
    if any(not isinstance(dim, Integral) or isinstance(dim, bool) for dim in shape[:2]):
        raise ValueError("image_shape dimensions must be integers")
    result = features.copy()
    if result.empty:
        result["nearest_neighbor_distance_px"] = pd.Series(dtype=float)
        result["x_norm"] = pd.Series(dtype=float)
        result["y_norm"] = pd.Series(dtype=float)
        result["cell_density_per_100k_px"] = pd.Series(dtype=float)
        return result
    h, w = float(shape[0]), float(shape[1])
    result["nearest_neighbor_distance_px"] = nearest_neighbor_distances(result)
    # Pixel coordinates span 0..w-1 and 0..h-1; use those extents so the
    # normalized coordinates genuinely map the image bounds to [0, 1].
    result["x_norm"] = result["centroid_x"] / max(w - 1.0, 1.0)
    result["y_norm"] = result["centroid_y"] / max(h - 1.0, 1.0)
    result["cell_density_per_100k_px"] = len(result) / (h * w) * 100000.0
    return result


def summarize_spatial_features(features: pd.DataFrame, image_shape: Sequence[int]) -> dict[str, float]:
    """Return dataset-level spatial statistics."""
    enriched = add_spatial_features(features, image_shape)
    nn = enriched["nearest_neighbor_distance_px"].dropna().to_numpy(dtype=float)
    h, w = float(image_shape[0]), float(image_shape[1])
    return {
        "object_count": float(len(enriched)),
        "density_per_100k_px": float(len(enriched) / (h * w) * 100000.0),
        "mean_nearest_neighbor_px": float(nn.mean()) if nn.size else float("nan"),
        "median_nearest_neighbor_px": float(np.median(nn)) if nn.size else float("nan"),
        "nearest_neighbor_cv": float(nn.std(ddof=1) / nn.mean()) if nn.size > 1 and nn.mean() else np.nan,
    }


def channel_summary(image: np.ndarray) -> pd.DataFrame:
    """Summarize each channel in a 2-D or HxWxC image.

    For integer images, saturation fractions are the fractions at the dtype
    endpoints. Floating-point images have no universal detector saturation
    range, so those fields are reported as NaN rather than treating zero as
    saturation.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or 0 in arr.shape:
        raise ValueError("channel_summary expects a non-empty 2-D or HxWxC image")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("image must have a numeric dtype")
    rows = []
    for channel in range(arr.shape[2]):
        values = arr[:, :, channel].astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("image values must be finite")
        lo, hi = np.percentile(values, [1, 99])
        if np.issubdtype(arr.dtype, np.integer):
            dtype_info = np.iinfo(arr.dtype)
            low_fraction = float((values <= dtype_info.min).mean())
            high_fraction = float((values >= dtype_info.max).mean())
        else:
            low_fraction = high_fraction = float("nan")
        rows.append({"channel": channel, "mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()), "max": float(values.max()), "p01": float(lo), "p99": float(hi), "saturation_low_fraction": low_fraction, "saturation_high_fraction": high_fraction})
    return pd.DataFrame(rows)


def object_channel_intensity(image: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Measure mean/median/max intensity for every object in every channel."""
    arr = np.asarray(image)
    lab = np.asarray(labels)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or lab.shape != arr.shape[:2]:
        raise ValueError("image must be HxW or HxWxC and labels must match HxW")
    if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
        raise ValueError("image must have a finite numeric dtype")
    if lab.ndim != 2 or not np.issubdtype(lab.dtype, np.integer):
        raise ValueError("labels must be a 2-D integer array")
    if (lab < 0).any():
        raise ValueError("labels must be non-negative")
    label_ids = np.unique(lab)
    label_ids = label_ids[label_ids > 0]
    rows = []
    for label_id in label_ids:
        mask = lab == label_id
        for channel in range(arr.shape[2]):
            values = arr[:, :, channel][mask].astype(np.float64, copy=False)
            if values.size == 0:
                continue
            rows.append({"label": int(label_id), "channel": int(channel), "mean_intensity": float(values.mean()), "median_intensity": float(np.median(values)), "std_intensity": float(values.std()), "max_intensity": float(values.max()), "integrated_intensity": float(values.sum())})
    return pd.DataFrame(rows)


def colocated_fraction(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Fraction of A's positive pixels overlapping B's positive pixels."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Masks must have identical shapes")
    total = int(a.sum())
    return float((a & b).sum() / total) if total else 0.0


def normalized_colocalization(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Return Pearson correlation between two finite, same-sized channels."""
    a = np.asarray(image_a, dtype=np.float64).ravel()
    b = np.asarray(image_b, dtype=np.float64).ravel()
    if a.size != b.size or a.size == 0:
        raise ValueError("Images must contain the same non-zero number of pixels")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Images must contain only finite values")
    a_std = a.std()
    b_std = b.std()
    if a_std == 0 or b_std == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def apply_background_correction(gray: np.ndarray, radius: int = 25) -> np.ndarray:
    """Subtract a smooth morphological background while retaining uint8 output."""
    values = np.asarray(gray)
    if values.ndim != 2 or values.dtype != np.uint8 or values.size == 0:
        raise ValueError("apply_background_correction expects a non-empty 2-D uint8 image")
    if not isinstance(radius, Integral) or isinstance(radius, bool) or radius < 1:
        raise ValueError("radius must be a positive integer")
    size = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    background = cv2.morphologyEx(values, cv2.MORPH_OPEN, kernel)
    corrected = cv2.subtract(values, background)
    return corrected
