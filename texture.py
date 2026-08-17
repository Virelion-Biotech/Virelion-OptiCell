"""Texture and heterogeneity features for microscopy phenotyping."""
from __future__ import annotations

import numpy as np
import pandas as pd


def basic_texture_features(gray: np.ndarray) -> dict[str, float]:
    """Dependency-light texture descriptors for a 2-D grayscale image."""
    arr = np.asarray(gray, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("gray must be a non-empty 2-D image")
    values = arr.ravel()
    hist, _ = np.histogram(values, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum())
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


def object_texture_features(gray: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Compute compact texture/heterogeneity features per segmented object."""
    image = np.asarray(gray, dtype=np.float32)
    masks = np.asarray(labels)
    if image.ndim != 2 or masks.shape != image.shape:
        raise ValueError("gray and labels must be matching 2-D arrays")
    rows = []
    for label in np.unique(masks):
        if label <= 0:
            continue
        mask = masks == label
        values = image[mask]
        if values.size == 0:
            continue
        gy, gx = np.gradient(image)
        grad = np.hypot(gx[mask], gy[mask])
        rows.append({
            "label": int(label),
            "texture_entropy": basic_texture_features(values.reshape(-1, 1))["intensity_entropy"],
            "texture_intensity_std": float(values.std()),
            "texture_p10": float(np.percentile(values, 10)),
            "texture_p90": float(np.percentile(values, 90)),
            "texture_gradient_mean": float(grad.mean()) if grad.size else 0.0,
            "texture_gradient_std": float(grad.std()) if grad.size else 0.0,
        })
    return pd.DataFrame(rows)
