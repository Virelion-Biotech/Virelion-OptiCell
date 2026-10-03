"""Cell/nucleus compartment analysis utilities for OptiCell."""
from __future__ import annotations

from typing import Optional
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def segment_nuclei(gray: np.ndarray, min_area: int = 20, max_area_frac: float = 0.15, adaptive: bool = False) -> np.ndarray:
    """Return relabeled nuclear instances using conservative thresholding."""
    image = np.asarray(gray)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("segment_nuclei expects a 2-D uint8 image")
    if min_area < 1 or not np.isfinite(max_area_frac) or not 0 < max_area_frac <= 1:
        raise ValueError("invalid nucleus area limits")
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    if adaptive:
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    else:
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float((binary > 0).mean()) > 0.5:
        binary = cv2.bitwise_not(binary)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    max_area = image.shape[0] * image.shape[1] * max_area_frac
    out = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            out[labels == i] = next_id
            next_id += 1
    return out


def assign_nuclei_to_cells(cell_labels: np.ndarray, nucleus_labels: np.ndarray, max_distance_px: Optional[float] = None) -> pd.DataFrame:
    """Assign nuclei to cells by centroid containment, then nearest cell pixel."""
    cells = np.asarray(cell_labels)
    nuclei = np.asarray(nucleus_labels)
    if cells.ndim != 2 or nuclei.ndim != 2 or cells.shape != nuclei.shape:
        raise ValueError("cell_labels and nucleus_labels must be matching 2-D arrays")
    if not np.issubdtype(cells.dtype, np.integer) or not np.issubdtype(nuclei.dtype, np.integer):
        raise ValueError("cell_labels and nucleus_labels must contain integer instance IDs")
    cells = cells.astype(np.int32, copy=False)
    nuclei = nuclei.astype(np.int32, copy=False)
    if max_distance_px is not None and (not np.isfinite(max_distance_px) or max_distance_px < 0):
        raise ValueError("max_distance_px must be a finite non-negative value")

    cell_pixels = np.argwhere(cells > 0)
    cell_tree = cKDTree(cell_pixels[:, ::-1]) if cell_pixels.size else None  # x,y coordinates
    rows = []
    for nid in np.unique(nuclei):
        if nid <= 0:
            continue
        y, x = np.nonzero(nuclei == nid)
        if not len(x):
            continue
        cx, cy = float(x.mean()), float(y.mean())
        ix = int(np.clip(np.rint(cx), 0, cells.shape[1] - 1))
        iy = int(np.clip(np.rint(cy), 0, cells.shape[0] - 1))
        parent = int(cells[iy, ix])
        distance = 0.0 if parent > 0 else np.inf
        if parent == 0 and cell_tree is not None:
            distance, nearest = cell_tree.query([[cx, cy]], k=1)
            distance = float(distance[0])
            py, px = cell_pixels[int(nearest[0])]
            parent = int(cells[py, px])
        if max_distance_px is not None and distance > max_distance_px:
            parent = 0
        rows.append({"nucleus_label": int(nid), "cell_label": parent, "nucleus_area_px": int(len(x)), "nucleus_centroid_x": cx, "nucleus_centroid_y": cy, "assignment_distance_px": distance})
    return pd.DataFrame(rows)


def compartment_features(image: np.ndarray, cell_labels: np.ndarray, nucleus_labels: np.ndarray, channel: int = 0) -> pd.DataFrame:
    """Calculate nucleus/cytoplasm area and intensity features for each assigned cell."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError("image must contain numeric values")
        if not 0 <= channel < arr.shape[2]:
            raise IndexError("channel outside image range")
        intensity = arr[:, :, channel]
    elif arr.ndim == 2:
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError("image must contain numeric values")
        intensity = arr
    else:
        raise ValueError("image must be 2-D or HxWxC")
    cells = np.asarray(cell_labels)
    nuclei = np.asarray(nucleus_labels)
    if cells.ndim != 2 or nuclei.ndim != 2 or cells.shape != intensity.shape or nuclei.shape != intensity.shape:
        raise ValueError("image and label arrays must have identical 2-D spatial shapes")
    assignments = assign_nuclei_to_cells(cells, nuclei)
    rows = []
    for cid in np.unique(cells):
        if cid <= 0:
            continue
        cell_mask = cells == cid
        nucleus_ids = assignments.loc[assignments["cell_label"] == cid, "nucleus_label"].astype(int).tolist() if not assignments.empty else []
        nucleus_mask = np.isin(nuclei, nucleus_ids)
        cyto_mask = cell_mask & ~nucleus_mask
        cell_values = intensity[cell_mask].astype(float)
        nuc_values = intensity[nucleus_mask].astype(float)
        cyto_values = intensity[cyto_mask].astype(float)
        cell_area = int(cell_mask.sum())
        nucleus_area = int(nucleus_mask.sum())
        cyto_area = int(cyto_mask.sum())
        nuc_mean = float(nuc_values.mean()) if nuc_values.size else np.nan
        cyto_mean = float(cyto_values.mean()) if cyto_values.size else np.nan
        rows.append({
            "cell_label": int(cid), "nucleus_count": len(nucleus_ids), "cell_area_px": cell_area,
            "nucleus_area_px": nucleus_area, "cytoplasm_area_px": cyto_area,
            "nucleus_to_cell_area_ratio": nucleus_area / cell_area if cell_area else np.nan,
            "cell_mean_intensity": float(cell_values.mean()) if cell_values.size else np.nan,
            "nucleus_mean_intensity": nuc_mean, "cytoplasm_mean_intensity": cyto_mean,
            "nucleus_cytoplasm_intensity_ratio": nuc_mean / cyto_mean if np.isfinite(cyto_mean) and cyto_mean != 0 else np.nan,
        })
    return pd.DataFrame(rows)
