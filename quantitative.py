"""Higher-level quantitative microscopy analysis utilities.

These functions operate on OptiCell segmentation outputs and raw NumPy image
arrays. They intentionally remain GUI-free and dependency-light.
"""

from __future__ import annotations

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
    n = len(points)
    if n < 2:
        return np.full(n, np.nan, dtype=float)
    # Chunked pairwise distances avoids a single huge NxN allocation.
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
    if len(image_shape) < 2 or image_shape[0] <= 0 or image_shape[1] <= 0:
        raise ValueError("image_shape must contain positive height and width")
    result = features.copy()
    if result.empty:
        result["nearest_neighbor_distance_px"] = pd.Series(dtype=float)
        result["x_norm"] = pd.Series(dtype=float)
        result["y_norm"] = pd.Series(dtype=float)
        return result
    h, w = float(image_shape[0]), float(image_shape[1])
    result["nearest_neighbor_distance_px"] = nearest_neighbor_distances(result)
    result["x_norm"] = result["centroid_x"] / w
    result["y_norm"] = result["centroid_y"] / h
    result["cell_density_per_100k_px"] = len(result) / (h * w) * 100000.0
    return result


def summarize_spatial_features(features: pd.DataFrame, image_shape: Sequence[int]) -> dict[str, float]:
    """Return dataset-level spatial statistics."""
    enriched = add_spatial_features(features, image_shape)
    nn = enriched["nearest_neighbor_distance_px"].dropna().to_numpy(dtype=float)
    h, w = float(image_shape[0]), float(image_shape[1])
    return {
        "object_count": float(len(enriched)),
        "density_per_100k_px": float(len(enriched) / (h * w) * 100000.0) if h and w else 0.0,
        "mean_nearest_neighbor_px": float(nn.mean()) if nn.size else float("nan"),
        "median_nearest_neighbor_px": float(np.median(nn)) if nn.size else float("nan"),
        "nearest_neighbor_cv": float(nn.std(ddof=1) / nn.mean()) if nn.size > 1 and nn.mean() else 0.0,
    }


def channel_summary(image: np.ndarray) -> pd.DataFrame:
    """Summarize each channel in a 2-D or HxWxC image."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError("channel_summary expects a 2-D grayscale or HxWxC image")
    rows = []
    for channel in range(arr.shape[2]):
        values = arr[:, :, channel].astype(np.float32, copy=False)
        lo, hi = np.percentile(values, [1, 99]) if values.size else (0.0, 0.0)
        rows.append(
            {
                "channel": channel,
                "mean": float(values.mean()) if values.size else 0.0,
                "std": float(values.std()) if values.size else 0.0,
                "min": float(values.min()) if values.size else 0.0,
                "max": float(values.max()) if values.size else 0.0,
                "p01": float(lo),
                "p99": float(hi),
                "saturation_low_fraction": float((values <= np.iinfo(arr.dtype).min).mean()) if np.issubdtype(arr.dtype, np.integer) else float((values <= 0).mean()),
                "saturation_high_fraction": float((values >= np.iinfo(arr.dtype).max).mean()) if np.issubdtype(arr.dtype, np.integer) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def object_channel_intensity(image: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Measure mean/median/max intensity for every object in every channel."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or labels.shape != arr.shape[:2]:
        raise ValueError("image must be HxW or HxWxC and labels must match HxW")
    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids > 0]
    rows = []
    for label_id in label_ids:
        mask = labels == label_id
        for channel in range(arr.shape[2]):
            values = arr[:, :, channel][mask].astype(np.float32, copy=False)
            if values.size == 0:
                continue
            rows.append(
                {
                    "label": int(label_id),
                    "channel": int(channel),
                    "mean_intensity": float(values.mean()),
                    "median_intensity": float(np.median(values)),
                    "std_intensity": float(values.std()),
                    "max_intensity": float(values.max()),
                    "integrated_intensity": float(values.sum()),
                }
            )
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
    """Pearson correlation between two channels, guarded against zero variance."""
    a = np.asarray(image_a, dtype=np.float64).ravel()
    b = np.asarray(image_b, dtype=np.float64).ravel()
    if a.size != b.size or a.size == 0:
        raise ValueError("Images must contain the same non-zero number of pixels")
    a_std = a.std()
    b_std = b.std()
    if a_std == 0 or b_std == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def apply_background_correction(gray: np.ndarray, radius: int = 25) -> np.ndarray:
    """Subtract a smooth morphological background while retaining uint8 output."""
    values = np.asarray(gray)
    if values.ndim != 2 or values.dtype != np.uint8:
        raise ValueError("apply_background_correction expects a 2-D uint8 image")
    if radius < 1:
        raise ValueError("radius must be >= 1")
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    background = cv2.morphologyEx(values, cv2.MORPH_OPEN, kernel)
    corrected = cv2.subtract(values, background)
    return corrected
