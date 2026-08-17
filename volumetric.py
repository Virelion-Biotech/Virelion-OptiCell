"""3-D morphology and spatial analysis for labelled microscopy volumes."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage


def _validate_labels(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 3:
        raise ValueError("labels must be a 3-D array")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("labels must contain integer instance IDs")
    return arr


def volume_features(labels: np.ndarray, voxel_size: Sequence[float] = (1.0, 1.0, 1.0)) -> list[dict[str, float]]:
    """Return per-object volume, centroid, bounding box, and surface-area approximation."""
    arr = _validate_labels(labels)
    vz, vy, vx = (float(x) for x in voxel_size)
    if min(vz, vy, vx) <= 0:
        raise ValueError("voxel_size values must be positive")
    spacing = np.array([vz, vy, vx], dtype=float)
    rows: list[dict[str, float]] = []
    for label_id in np.unique(arr):
        if label_id <= 0:
            continue
        coords = np.argwhere(arr == label_id)
        if coords.size == 0:
            continue
        centroid_vox = coords.mean(axis=0)
        scaled = coords * spacing
        volume = float(len(coords) * vz * vy * vx)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        eroded = ndimage.binary_erosion(arr == label_id, structure=np.ones((3, 3, 3)), border_value=0)
        surface_voxels = int(np.logical_and(arr == label_id, ~eroded).sum())
        surface_area_approx = float(surface_voxels * min(vz * vy, vz * vx, vy * vx))
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
                "surface_area_approx": surface_area_approx,
            }
        )
    return rows


def nearest_neighbor_distances_3d(features: list[dict[str, float]]) -> np.ndarray:
    """Return nearest-neighbour centroid distances using physical coordinates."""
    if not features:
        return np.array([], dtype=float)
    points = np.asarray([[x["centroid_z_um"], x["centroid_y_um"], x["centroid_x_um"]] for x in features], dtype=float)
    if len(points) < 2:
        return np.full(len(points), np.nan, dtype=float)
    nearest = np.full(len(points), np.inf, dtype=float)
    for start in range(0, len(points), 512):
        stop = min(start + 512, len(points))
        block = points[start:stop]
        d2 = ((block[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        local = np.arange(start, stop)
        d2[np.arange(stop - start), local] = np.inf
        nearest[start:stop] = np.sqrt(d2.min(axis=1))
    return nearest


def summarize_volume(labels: np.ndarray, voxel_size: Sequence[float] = (1.0, 1.0, 1.0)) -> dict[str, float]:
    """Return object count, volume statistics, and physical density for a labelled volume."""
    arr = _validate_labels(labels)
    vz, vy, vx = (float(x) for x in voxel_size)
    if min(vz, vy, vx) <= 0:
        raise ValueError("voxel_size values must be positive")
    features = volume_features(arr, voxel_size)
    values = np.asarray([row["volume"] for row in features], dtype=float)
    physical_volume = float(np.prod(arr.shape) * vz * vy * vx)
    return {
        "object_count": float(len(features)),
        "total_object_volume": float(values.sum()) if values.size else 0.0,
        "mean_object_volume": float(values.mean()) if values.size else np.nan,
        "median_object_volume": float(np.median(values)) if values.size else np.nan,
        "volume_fraction": float(values.sum() / physical_volume) if physical_volume else 0.0,
        "object_density_per_mm3": float(len(features) / (physical_volume / 1e9)) if physical_volume else 0.0,
    }
