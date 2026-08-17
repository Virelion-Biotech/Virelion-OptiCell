"""OptiCell microscopy analysis engine."""

from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np
import pandas as pd

try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:  # pragma: no cover
    tifffile = None
    _HAS_TIFFFILE = False

try:
    from cellpose import models as _cellpose_models
    _HAS_CELLPOSE = True
except ImportError:  # pragma: no cover
    _cellpose_models = None
    _HAS_CELLPOSE = False

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
PIPELINE_VERSION = "2.0.0"


@dataclass(frozen=True)
class QCThresholds:
    focus_min: float = 100.0
    brightness_min: float = 25.0
    brightness_max: float = 230.0
    saturation_max_fraction: float = 0.02
    min_cell_area: int = 15
    max_cell_area_frac: float = 0.25
    cell_count_low: int = 1
    cell_count_high: Optional[int] = None

    def validate(self) -> None:
        if self.focus_min < 0:
            raise ValueError("focus_min must be >= 0")
        if not 0 <= self.brightness_min <= 255:
            raise ValueError("brightness_min must be in [0, 255]")
        if not 0 <= self.brightness_max <= 255 or self.brightness_max < self.brightness_min:
            raise ValueError("brightness_max must be in [brightness_min, 255]")
        if not 0 <= self.saturation_max_fraction <= 1:
            raise ValueError("saturation_max_fraction must be in [0, 1]")
        if self.min_cell_area < 1:
            raise ValueError("min_cell_area must be >= 1")
        if not 0 < self.max_cell_area_frac <= 1:
            raise ValueError("max_cell_area_frac must be in (0, 1]")
        if self.cell_count_low < 0:
            raise ValueError("cell_count_low must be >= 0")
        if self.cell_count_high is not None and self.cell_count_high < self.cell_count_low:
            raise ValueError("cell_count_high must be >= cell_count_low")


@dataclass
class SegmentationResult:
    count: int
    labels: np.ndarray
    method: str
    foreground_fraction: float
    median_area: float
    area_cv: float
    border_fraction: float
    tiny_object_fraction: float
    merged_object_fraction: float
    quality_score: float
    error: Optional[str] = None


@dataclass
class ImageResult:
    filename: str
    path: str
    width: int
    height: int
    channels: int
    dtype: str
    ndim: int
    file_size_kb: float
    sha256: str
    focus_score: float
    brightness_mean: float
    brightness_std: float
    saturation_fraction: float
    contrast_std: float
    estimated_cells: int
    cell_method: str
    segmentation_quality: float
    median_cell_area: float
    cell_area_cv: float
    border_object_fraction: float
    flags: list[str] = field(default_factory=list)
    adaptive_score: Optional[float] = None
    error: Optional[str] = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["flags"] = "; ".join(self.flags)
        return row


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _collapse_tiff_stack(arr: np.ndarray) -> np.ndarray:
    if arr.ndim <= 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return arr
    while arr.ndim > 3:
        arr = np.max(arr, axis=0)
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = np.max(arr, axis=0)
    return arr


def load_image(path: str) -> np.ndarray:
    path = os.fspath(path)
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {ext}")
    if ext in (".tif", ".tiff") and _HAS_TIFFFILE:
        return _collapse_tiff_stack(np.asarray(tifffile.imread(path)))
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read image: {path}")
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    return arr


def _rescale_to_uint8(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("Image array must contain numeric values")
    work = values.astype(np.float32, copy=False)
    if work.size == 0:
        return np.zeros(work.shape, dtype=np.uint8)
    work = np.nan_to_num(work, copy=False)
    if work.ndim == 3:
        flat = work.reshape(-1, work.shape[-1])
    else:
        flat = work.reshape(-1, 1)
    sample = flat[:: max(1, len(flat) // 200000)]
    lo = np.percentile(sample, 0.5, axis=0)
    hi = np.percentile(sample, 99.5, axis=0)
    if work.ndim == 3:
        lo_b = lo.reshape(1, 1, -1)
        hi_b = hi.reshape(1, 1, -1)
        span = np.where((hi_b - lo_b) > 0, hi_b - lo_b, 1.0)
        scaled = np.clip((work - lo_b) / span, 0, 1) * 255.0
    else:
        lo_v, hi_v = float(lo[0]), float(hi[0])
        if hi_v <= lo_v:
            lo_v, hi_v = float(work.min()), float(work.max())
        if hi_v <= lo_v:
            return np.zeros(work.shape, dtype=np.uint8)
        scaled = np.clip((work - lo_v) / (hi_v - lo_v), 0, 1) * 255.0
    return scaled.astype(np.uint8)


def to_grayscale_uint8(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr)
    if values.ndim == 2:
        return values if values.dtype == np.uint8 else _rescale_to_uint8(values)
    if values.ndim != 3:
        raise ValueError(f"Expected 2-D or 3-D image, got shape {values.shape}")
    if values.shape[2] == 1:
        return to_grayscale_uint8(values[:, :, 0])
    rgb8 = values[:, :, :3] if values.dtype == np.uint8 else _rescale_to_uint8(values[:, :, :3])
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)


def compute_focus_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(gray: np.ndarray) -> tuple[float, float]:
    return float(gray.mean()), float(gray.std())


def compute_saturation_fraction(gray: np.ndarray, low: int = 1, high: int = 254) -> float:
    if gray.size == 0:
        return 0.0
    return float(((gray <= low) | (gray >= high)).mean())


def _safe_cv(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    mean = float(arr.mean())
    return float(arr.std(ddof=1) / mean) if mean else 0.0


def _segmentation_diagnostics(labels: np.ndarray, image_shape: tuple[int, int], min_area: int, max_area_frac: float) -> tuple[float, float, float, float, float, float, float]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((labels > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    total_pixels = float(image_shape[0] * image_shape[1])
    foreground_fraction = float(areas.sum() / total_pixels) if total_pixels else 0.0
    median_area = float(np.median(areas))
    area_cv = _safe_cv(areas)
    border_count = 0
    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT]); y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH]); h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if x == 0 or y == 0 or x + w >= image_shape[1] or y + h >= image_shape[0]:
            border_count += 1
    object_count = max(1, num_labels - 1)
    border_fraction = float(border_count / object_count)
    tiny_fraction = float((areas < max(1, min_area * 2)).mean())
    merged_fraction = float((areas > total_pixels * max_area_frac).mean()) if areas.size else 0.0
    quality = 100.0
    quality -= min(45.0, border_fraction * 30.0)
    quality -= min(30.0, tiny_fraction * 30.0)
    quality -= min(25.0, merged_fraction * 25.0)
    quality = float(np.clip(quality, 0.0, 100.0))
    return foreground_fraction, median_area, area_cv, border_fraction, tiny_fraction, merged_fraction, quality


def _build_segmentation_result(labels: np.ndarray, gray: np.ndarray, method: str, min_area: int, max_area_frac: float, error: Optional[str] = None) -> SegmentationResult:
    nonzero_labels = np.unique(labels[labels > 0]) if labels.size else np.asarray([], dtype=np.int32)
    count = int(nonzero_labels.size)
    foreground_fraction, median_area, area_cv, border_fraction, tiny_fraction, merged_fraction, quality = _segmentation_diagnostics(labels, gray.shape, min_area, max_area_frac)
    return SegmentationResult(count=count, labels=labels.astype(np.int32, copy=False), method=method, foreground_fraction=foreground_fraction, median_area=median_area, area_cv=area_cv, border_fraction=border_fraction, tiny_object_fraction=tiny_fraction, merged_object_fraction=merged_fraction, quality_score=quality, error=error)


def segment_threshold(gray: np.ndarray, min_area: int = 15, max_area_frac: float = 0.25, adaptive: bool = False) -> SegmentationResult:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    if adaptive:
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
    else:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float((thresh > 0).mean()) > 0.5:
        thresh = cv2.bitwise_not(thresh)
    fg = blurred[thresh > 0]; bg = blurred[thresh == 0]
    if fg.size == 0 or bg.size == 0 or abs(float(fg.mean()) - float(bg.mean())) < 8:
        labels = np.zeros_like(gray, dtype=np.int32)
        return _build_segmentation_result(labels, gray, "threshold", min_area, max_area_frac)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, raw_labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    max_area = gray.shape[0] * gray.shape[1] * max_area_frac
    keep = np.zeros_like(raw_labels, dtype=np.int32)
    next_id = 1
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            keep[raw_labels == label_id] = next_id
            next_id += 1
    return _build_segmentation_result(keep, gray, "threshold", min_area, max_area_frac)


class CellposeSegmenter:
    def __init__(self, model_type: str = "cyto3", gpu: bool | None = None) -> None:
        if not _HAS_CELLPOSE:
            raise RuntimeError("Cellpose is not installed. Install the optional cellpose dependency.")
        self.model_type = model_type; self.gpu = gpu; self._model = None
    @property
    def model(self):
        if self._model is None:
            kwargs: dict[str, Any] = {"model_type": self.model_type}
            if self.gpu is not None: kwargs["gpu"] = bool(self.gpu)
            try: self._model = _cellpose_models.CellposeModel(**kwargs)
            except AttributeError: self._model = _cellpose_models.Cellpose(**kwargs)
        return self._model
    def segment(self, gray: np.ndarray, diameter: Optional[float] = None, min_area: int = 15, max_area_frac: float = 0.25) -> SegmentationResult:
        result = self.model.eval(gray, diameter=diameter, channels=[0, 0]); masks = result[0]
        return _build_segmentation_result(np.asarray(masks, dtype=np.int32), gray, f"cellpose:{self.model_type}", min_area, max_area_frac)


def extract_object_features(gray: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = labels.astype(np.int32, copy=False)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats((labels > 0).astype(np.uint8), connectivity=8)
    for label_id in range(1, num_labels):
        mask = labels == label_id; area = int(mask.sum())
        if area <= 0: continue
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0
        circularity = float((4 * np.pi * area) / (perimeter * perimeter)) if perimeter else 0.0
        x = int(stats[label_id, cv2.CC_STAT_LEFT]); y = int(stats[label_id, cv2.CC_STAT_TOP]); w = int(stats[label_id, cv2.CC_STAT_WIDTH]); h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        values = gray[mask]
        rows.append({"label": label_id, "area_px": area, "perimeter_px": perimeter, "circularity": float(np.clip(circularity, 0, 1)), "bbox_x": x, "bbox_y": y, "bbox_width": w, "bbox_height": h, "aspect_ratio": float(w / h) if h else 0.0, "centroid_x": float(centroids[label_id][0]), "centroid_y": float(centroids[label_id][1]), "mean_intensity": float(values.mean()), "std_intensity": float(values.std()), "max_intensity": int(values.max())})
    return pd.DataFrame(rows)


def robust_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce"); median = float(values.median()); mad = float((values - median).abs().median())
    if not np.isfinite(mad) or mad == 0: return pd.Series(np.zeros(len(values)), index=series.index, dtype=float)
    return 0.6745 * (values - median) / mad


def adaptive_dataset_qc(df: pd.DataFrame, z_limit: float = 3.5) -> pd.DataFrame:
    result = df.copy()
    if result.empty: result["adaptive_score"] = pd.Series(dtype=float); return result
    metric_cols = ["focus_score", "brightness_mean", "saturation_fraction", "contrast_std", "estimated_cells", "segmentation_quality"]
    z_columns: list[str] = []
    for col in metric_cols:
        if col in result.columns:
            zcol = f"{col}_robust_z"; result[zcol] = robust_zscore(result[col]); z_columns.append(zcol)
    result["adaptive_score"] = result[z_columns].abs().max(axis=1).fillna(0.0) if z_columns else 0.0
    existing = result.get("flags", pd.Series("", index=result.index)).fillna("").astype(str)
    is_outlier = result["adaptive_score"] >= z_limit
    result.loc[is_outlier, "flags"] = [f"{flag}; ADAPTIVE_OUTLIER".strip("; ") if "ADAPTIVE_OUTLIER" not in flag else flag for flag in existing.loc[is_outlier]]
    return result


def _empty_result(path: str, error: str, requested_method: str) -> ImageResult:
    exists = os.path.exists(path); file_size = os.path.getsize(path) / 1024.0 if exists else 0.0; digest = sha256_file(path) if exists else ""
    return ImageResult(filename=os.path.basename(path), path=os.path.abspath(path), width=0, height=0, channels=0, dtype="unknown", ndim=0, file_size_kb=round(file_size, 2), sha256=digest, focus_score=0.0, brightness_mean=0.0, brightness_std=0.0, saturation_fraction=0.0, contrast_std=0.0, estimated_cells=0, cell_method=requested_method, segmentation_quality=0.0, median_cell_area=0.0, cell_area_cv=0.0, border_object_fraction=0.0, flags=["FAILED_TO_LOAD"], error=error)


def analyze_image(path: str, thresholds: Optional[QCThresholds] = None, cell_method: str = "threshold", cellpose_segmenter: Optional[CellposeSegmenter] = None, adaptive_threshold: bool = False, return_segmentation: bool = False) -> ImageResult | tuple[ImageResult, SegmentationResult]:
    thresholds = thresholds or QCThresholds(); thresholds.validate(); requested_method = cell_method.lower()
    if requested_method not in {"threshold", "cellpose"}: raise ValueError("cell_method must be 'threshold' or 'cellpose'")
    try:
        raw = load_image(path); gray = to_grayscale_uint8(raw)
    except Exception as exc:
        result = _empty_result(path, str(exc), requested_method); return (result, _build_segmentation_result(np.zeros((1, 1), dtype=np.int32), np.zeros((1, 1), dtype=np.uint8), requested_method, thresholds.min_cell_area, thresholds.max_cell_area_frac)) if return_segmentation else result
    focus_score = compute_focus_score(gray); brightness_mean, brightness_std = compute_brightness(gray); saturation_fraction = compute_saturation_fraction(gray); contrast_std = brightness_std
    if requested_method == "threshold": segmentation = segment_threshold(gray, thresholds.min_cell_area, thresholds.max_cell_area_frac, adaptive_threshold)
    else:
        segmenter = cellpose_segmenter or CellposeSegmenter(gpu=None); segmentation = segmenter.segment(gray, min_area=thresholds.min_cell_area, max_area_frac=thresholds.max_cell_area_frac)
    flags: list[str] = []
    if focus_score < thresholds.focus_min: flags.append("BLURRY")
    if brightness_mean < thresholds.brightness_min: flags.append("TOO_DARK")
    if brightness_mean > thresholds.brightness_max: flags.append("TOO_BRIGHT")
    if saturation_fraction > thresholds.saturation_max_fraction: flags.append("SATURATION")
    if segmentation.count < thresholds.cell_count_low: flags.append("LOW_CELL_COUNT")
    if thresholds.cell_count_high is not None and segmentation.count > thresholds.cell_count_high: flags.append("HIGH_CELL_COUNT")
    result = ImageResult(filename=os.path.basename(path), path=os.path.abspath(path), width=int(gray.shape[1]), height=int(gray.shape[0]), channels=int(raw.shape[2]) if raw.ndim == 3 else 1, dtype=str(raw.dtype), ndim=int(raw.ndim), file_size_kb=round(os.path.getsize(path) / 1024.0, 2), sha256=sha256_file(path), focus_score=focus_score, brightness_mean=brightness_mean, brightness_std=brightness_std, saturation_fraction=saturation_fraction, contrast_std=contrast_std, estimated_cells=segmentation.count, cell_method=segmentation.method, segmentation_quality=segmentation.quality_score, median_cell_area=segmentation.median_area, cell_area_cv=segmentation.area_cv, border_object_fraction=segmentation.border_fraction, flags=flags)
    return (result, segmentation) if return_segmentation else result
