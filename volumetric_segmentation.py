"""Deterministic 3-D threshold segmentation for microscopy volumes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class VolumetricSegmentationResult:
    labels: np.ndarray
    method: str
    count: int
    foreground_fraction: float
    median_volume_voxels: float
    volume_cv: float


def _validate_volume(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError("volume must be a 3-D array")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError("volume must contain numeric values")
    if arr.size == 0:
        raise ValueError("volume cannot be empty")
    return arr


def _otsu_threshold(values: np.ndarray) -> float:
    """Compute a normalized Otsu threshold without adding another dependency."""
    hist, edges = np.histogram(values.ravel(), bins=256, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return 0.5
    probabilities = hist.astype(float) / total
    centers = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(probabilities)
    mu = np.cumsum(probabilities * centers)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    sigma = np.where(denom > 0, (mu_total * omega - mu) ** 2 / denom, 0.0)
    return float(centers[int(np.argmax(sigma))])


def segment_threshold_3d(
    volume: np.ndarray,
    voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
    min_volume_voxels: int = 20,
    connectivity: int = 1,
) -> VolumetricSegmentationResult:
    """Segment a 3-D volume using percentile normalization and connected components."""
    arr = _validate_volume(volume)
    spacing = tuple(float(v) for v in voxel_size)
    if len(spacing) != 3 or any(v <= 0 for v in spacing):
        raise ValueError("voxel_size must contain three positive values")
    if min_volume_voxels < 1:
        raise ValueError("min_volume_voxels must be >= 1")
    if connectivity not in {1, 2, 3}:
        raise ValueError("connectivity must be 1, 2, or 3")

    work = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(work, [0.5, 99.5])
    if hi <= lo:
        labels = np.zeros(work.shape, dtype=np.int32)
        return VolumetricSegmentationResult(labels, "threshold_3d", 0, 0.0, np.nan, 0.0)

    scaled = np.clip((work - lo) / (hi - lo), 0.0, 1.0)
    threshold = _otsu_threshold(scaled)
    mask = scaled >= threshold

    structure = ndimage.generate_binary_structure(3, connectivity)
    mask = ndimage.binary_opening(mask, structure=structure)
    mask = ndimage.binary_closing(mask, structure=structure)
    labels, count = ndimage.label(mask, structure=structure)

    out = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    volumes = []
    for label_id in range(1, count + 1):
        n = int((labels == label_id).sum())
        if n >= min_volume_voxels:
            out[labels == label_id] = next_id
            volumes.append(n)
            next_id += 1

    values = np.asarray(volumes, dtype=float)
    mean = float(values.mean()) if values.size else 0.0
    cv = float(values.std(ddof=1) / mean) if values.size > 1 and mean else 0.0
    return VolumetricSegmentationResult(
        labels=out,
        method="threshold_3d",
        count=len(volumes),
        foreground_fraction=float((out > 0).mean()),
        median_volume_voxels=float(np.median(values)) if values.size else np.nan,
        volume_cv=cv,
    )
