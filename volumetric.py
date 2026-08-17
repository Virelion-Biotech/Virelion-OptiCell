"""3-D morphology and spatial analysis for labelled microscopy volumes."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree


def _validate_labels(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 3:
        raise ValueError("labels must be a 3-D array")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("labels must contain integer instance IDs")
    return arr


def _surface_area(arr: np.ndarray, label_id: int, spacing: tuple[float, float, float]) -> float:
    """Compute exposed voxel-face area on a minimal local crop."""
    coords = np.argwhere(arr == label_id)
    if coords.size == 0:
        return 0.0
    mins = np.maximum(coords.min(axis=0) - 1, 0)
    maxs = np.minimum(coords.max(axis=0) + 2, np.asarray(arr.shape))
    z0, y0, x0 = mins
    z1, y1, x1 = maxs
    mask = arr[z0:z1, y0:y1, x0:x1] == label_id
    vz, vy, vx = spacing
    area = 0.0
    area += np.count_nonzero(mask[0, :, :]) * vy * vx
    area += np.count_nonzero(mask[-1, :, :]) * vy * vx
    area += np.count_nonzero(mask[:, 0, :]) * vz * vx
    area += np.count_nonzero(mask[:, -1, :]) * vz * vx
    area += np.count_nonzero(mask[:, :, 0]) * vz * vy
    area += np.count_nonzero(mask[:, :, -1]) * vz * vy
    area += np.count_nonzero(mask[1:, :, :] & ~mask[:-1, :, :]) * vy * vx
    area += np.count_nonzero(mask[:-1, :, :] & ~mask[1:, :, :]) * vy * vx
    area += np.count_nonzero(mask[:, 1:, :] & ~mask[:, :-1, :]) * vz * vx
    area += np.count_nonzero(mask[:, :-1, :] & ~mask[:, 1:, :]) * vz * vx
    area += np.count_nonzero(mask[:, :, 1:] & ~mask[:, :, :-1]) * vz * vy
    area += np.count_nonzero(mask[:, :, :-1] & ~mask[:, :, 1:]) * vz * vy
    return float(area)


def volume_features(labels: np.ndarray, voxel_size: Sequence[float] = (1.0, 1.0, 1.0)) -> list[dict[str, float]]:
    """Return per-object volume, centroid, bounding box, and surface area."""
    arr = _validate_labels(labels)
    spacing = tuple(float(x) for x in voxel_size)
    if len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError("voxel_size must contain three positive values")
    vz, vy, vx = spacing
    rows: list[dict[str, float]] = []
    for label_id in np.unique(arr):
        if label_id <= 0:
            continue
        coords = np.argwhere(arr == label_id)
        if coords.size == 0:
            continue
        centroid_vox = coords.mean(axis=0)
        volume = float(len(coords) * vz * vy * vx)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        rows.append(
            {
                "label": int(label_id),
                "volume": volume,
                "centroid_z": float(centroid_vox[0]),
                "centroid_y": float(centroid_vox[1]),
                "centroid_x": float(centroid_vox[2]),
                "centroid_z_um": float(centroid_vox[0] * vz),
                "centroid_y_um": float(centroid_vox[1] * vy),
                "centroid_x_um": float(centroid_vox[2] * vx),
                "bbox_z": int(maxs[0] - mins[0] + 1),
                "bbox_y": int(maxs[1] - mins[1] + 1),
                "bbox_x": int(maxs[2] - mins[2] + 1),
                "surface_area_approx": _surface_area(arr, int(label_id), spacing),
            }
        )
    return rows


def nearest_neighbor_distances_3d(features: list[dict[str, float]]) -> np.ndarray:
    """Return nearest-neighbour centroid distances using a KD-tree."""
    if not features:
        return np.array([], dtype=float)
    points = np.asarray([[x["centroid_z_um"], x["centroid_y_um"], x["centroid_x_um"]] for x in features], dtype=float)
    if len(points) < 2:
        return np.full(len(points), np.nan, dtype=float)
    distances, _ = cKDTree(points).query(points, k=2)
    return distances[:, 1].astype(float)


def summarize_volume(labels: np.ndarray, voxel_size: Sequence[float] = (1.0, 1.0, 1.0)) -> dict[str, float]:
    """Return object count, volume statistics, and physical density for a labelled volume."""
    arr = _validate_labels(labels)
    spacing = tuple(float(x) for x in voxel_size)
    if len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError("voxel_size must contain three positive values")
    features = volume_features(arr, spacing)
    values = np.asarray([row["volume"] for row in features], dtype=float)
    physical_volume = float(np.prod(arr.shape) * np.prod(spacing))
    return {
        "object_count": float(len(features)),
        "total_object_volume": float(values.sum()) if values.size else 0.0,
        "mean_object_volume": float(values.mean()) if values.size else np.nan,
        "median_object_volume": float(np.median(values)) if values.size else np.nan,
        "volume_fraction": float(values.sum() / physical_volume) if physical_volume else 0.0,
        "object_density_per_mm3": float(len(features) / (physical_volume / 1e9)) if physical_volume else 0.0,
    }
