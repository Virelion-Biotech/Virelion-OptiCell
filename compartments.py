"""Cell/nucleus compartment analysis utilities for OptiCell."""
from __future__ import annotations

from typing import Optional
import cv2
import numpy as np
import pandas as pd


def segment_nuclei(gray: np.ndarray, min_area: int = 20, max_area_frac: float = 0.15, adaptive: bool = False) -> np.ndarray:
    """Return relabeled nuclear instances using conservative thresholding."""
    image = np.asarray(gray)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("segment_nuclei expects a 2-D uint8 image")
    if min_area < 1 or not 0 < max_area_frac <= 1:
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
    """Assign nuclei to cells by nucleus centroid containment, then optional nearest cell."""
    cells = np.asarray(cell_labels, dtype=np.int32)
    nuclei = np.asarray(nucleus_labels, dtype=np.int32)
    if cells.shape != nuclei.shape or cells.ndim != 2:
        raise ValueError("cell_labels and nucleus_labels must be matching 2-D arrays")
    cell_ids = np.unique(cells); cell_ids = cell_ids[cell_ids > 0]
    rows = []
    centroids = {}
    for cid in cell_ids:
        y, x = np.nonzero(cells == cid)
        centroids[int(cid)] = (float(x.mean()), float(y.mean())) if len(x) else (np.nan, np.nan)
    for nid in np.unique(nuclei):
        if nid <= 0:
            continue
        y, x = np.nonzero(nuclei == nid)
        if not len(x):
            continue
        cx, cy = float(x.mean()), float(y.mean())
        parent = int(cells[int(round(cy)), int(round(cx))]) if 0 <= int(round(cy)) < cells.shape[0] and 0 <= int(round(cx)) < cells.shape[1] else 0
        distance = 0.0
        if parent == 0 and centroids:
            ids = np.asarray(list(centroids.keys()), dtype=int)
            pts = np.asarray([centroids[i] for i in ids])
            d = np.sqrt(((pts - np.array([cx, cy])) ** 2).sum(axis=1))
            idx = int(np.argmin(d)); distance = float(d[idx]); parent = int(ids[idx])
        if max_distance_px is not None and distance > max_distance_px:
            parent = 0
        rows.append({"nucleus_label": int(nid), "cell_label": parent, "nucleus_area_px": int(len(x)), "nucleus_centroid_x": cx, "nucleus_centroid_y": cy, "assignment_distance_px": distance})
    return pd.DataFrame(rows)


def compartment_features(image: np.ndarray, cell_labels: np.ndarray, nucleus_labels: np.ndarray, channel: int = 0) -> pd.DataFrame:
    """Calculate nucleus/cytoplasm area and intensity features for each assigned cell."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        if not 0 <= channel < arr.shape[2]:
            raise IndexError("channel outside image range")
        intensity = arr[:, :, channel]
    elif arr.ndim == 2:
        intensity = arr
    else:
        raise ValueError("image must be 2-D or HxWxC")
    cells = np.asarray(cell_labels, dtype=np.int32); nuclei = np.asarray(nucleus_labels, dtype=np.int32)
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
        cell_area = int(cell_mask.sum()); nucleus_area = int(nucleus_mask.sum()); cyto_area = int(cyto_mask.sum())
        nuc_mean = float(nuc_values.mean()) if nuc_values.size else np.nan
        cyto_mean = float(cyto_values.mean()) if cyto_values.size else np.nan
        rows.append({"cell_label": int(cid), "nucleus_count": len(nucleus_ids), "cell_area_px": cell_area, "nucleus_area_px": nucleus_area, "cytoplasm_area_px": cyto_area, "nucleus_to_cell_area_ratio": nucleus_area / cell_area if cell_area else np.nan, "cell_mean_intensity": float(cell_values.mean()) if cell_values.size else np.nan, "nucleus_mean_intensity": nuc_mean, "cytoplasm_mean_intensity": cyto_mean, "nucleus_cytoplasm_intensity_ratio": nuc_mean / cyto_mean if np.isfinite(cyto_mean) and cyto_mean != 0 else np.nan})
    return pd.DataFrame(rows)
