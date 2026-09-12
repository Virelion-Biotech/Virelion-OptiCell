"""OptiCell microscopy analysis engine.

Headless APIs for image QC, segmentation, object features, batch analysis,
and deterministic CSV/JSON export. Absolute QC flags are deliberately
heuristic and should not be interpreted as validated biological accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
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

_CELLPOSE_IMPORT_ERROR: Optional[str] = None
try:
    from cellpose import models as _cellpose_models
    _HAS_CELLPOSE = True
except Exception as exc:  # pragma: no cover
    _cellpose_models = None
    _HAS_CELLPOSE = False
    _CELLPOSE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
PIPELINE_VERSION = "2.0.4"


@dataclass(frozen=True)
class QCThresholds:
    """Absolute image-QC limits used alongside dataset-level adaptive QC."""

    focus_min: float = 100.0
    brightness_min: float = 25.0
    brightness_max: float = 230.0
    saturation_max_fraction: float = 0.02
    min_cell_area: int = 15
    max_cell_area_frac: float = 0.25
    cell_count_low: int = 1
    cell_count_high: Optional[int] = None

    def validate(self) -> None:
        finite = [self.focus_min, self.brightness_min, self.brightness_max, self.saturation_max_fraction, self.max_cell_area_frac]
        if any(not np.isfinite(float(v)) for v in finite):
            raise ValueError("QC thresholds must be finite")
        if self.focus_min < 0:
            raise ValueError("focus_min must be >= 0")
        if not 0 <= self.brightness_min <= 255:
            raise ValueError("brightness_min must be in [0, 255]")
        if not 0 <= self.brightness_max <= 255 or self.brightness_max < self.brightness_min:
            raise ValueError("brightness_max must be in [brightness_min, 255]")
        if not 0 <= self.saturation_max_fraction <= 1:
            raise ValueError("saturation_max_fraction must be in [0, 1]")
        if not isinstance(self.min_cell_area, (int, np.integer)) or isinstance(self.min_cell_area, bool) or self.min_cell_area < 1:
            raise ValueError("min_cell_area must be a positive integer")
        if not 0 < self.max_cell_area_frac <= 1:
            raise ValueError("max_cell_area_frac must be in (0, 1]")
        if not isinstance(self.cell_count_low, (int, np.integer)) or isinstance(self.cell_count_low, bool) or self.cell_count_low < 0:
            raise ValueError("cell_count_low must be a non-negative integer")
        if self.cell_count_high is not None and (
            not isinstance(self.cell_count_high, (int, np.integer))
            or isinstance(self.cell_count_high, bool)
            or self.cell_count_high < self.cell_count_low
        ):
            raise ValueError("cell_count_high must be a non-negative integer >= cell_count_low")


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
    if not isinstance(chunk_size, (int, np.integer)) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_tiff_shape(arr: np.ndarray) -> np.ndarray:
    """Accept only analysis-ready TIFF shapes; never infer Z/T/C semantics."""
    values = np.asarray(arr)
    if values.ndim <= 2:
        return values
    if values.ndim == 3 and values.shape[-1] in (3, 4):
        return values
    raise ValueError(
        "Multidimensional TIFFs must be explicitly reduced to a single 2-D plane "
        "or RGB/RGBA image before QC; no Z/T/C projection is inferred automatically"
    )


def load_image(path: str) -> np.ndarray:
    path = os.fspath(path)
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {ext}")
    if ext in (".tif", ".tiff") and _HAS_TIFFFILE:
        return _validate_tiff_shape(np.asarray(tifffile.imread(path)))
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read image: {path}")
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    elif arr.ndim > 3:
        raise ValueError(f"Expected 2-D or RGB/RGBA image, got shape {arr.shape}")
    return arr


def _rescale_to_uint8(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("Image array must contain numeric values")
    if not np.isfinite(values.astype(np.float64, copy=False)).all():
        raise ValueError("Image array must contain only finite numeric values")
    work = values.astype(np.float32, copy=False)
    if work.size == 0:
        return np.zeros(work.shape, dtype=np.uint8)
    flat = work.reshape(-1, work.shape[-1]) if work.ndim == 3 else work.reshape(-1, 1)
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
    if values.shape[2] not in (3, 4):
        raise ValueError(f"Expected grayscale or RGB/RGBA image, got shape {values.shape}")
    rgb8 = values[:, :, :3] if values.dtype == np.uint8 else _rescale_to_uint8(values[:, :, :3])
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)


def compute_focus_score(gray: np.ndarray) -> float:
    arr = np.asarray(gray)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("gray must be a non-empty 2-D image")
    if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr.astype(np.float64, copy=False)).all():
        raise ValueError("gray must contain finite numeric values")
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())


def compute_brightness(gray: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(gray)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("gray must be a non-empty 2-D image")
    if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr.astype(np.float64, copy=False)).all():
        raise ValueError("gray must contain finite numeric values")
    return float(arr.mean()), float(arr.std())


def compute_saturation_fraction(gray: np.ndarray, low: int = 1, high: int = 254) -> float:
    arr = np.asarray(gray)
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.number):
        raise ValueError("gray must be a 2-D numeric image")
    if not np.isfinite(float(low)) or not np.isfinite(float(high)) or low > high:
        raise ValueError("low/high must be finite with low <= high")
    if arr.size == 0:
        return 0.0
    if not np.isfinite(arr.astype(np.float64, copy=False)).all():
        raise ValueError("gray must contain finite numeric values")
    return float(((arr <= low) | (arr >= high)).mean())


def _safe_cv(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    mean = float(arr.mean())
    return float(arr.std(ddof=1) / mean) if mean else 0.0


def _instance_objects(labels: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Return each positive instance mask by its original integer label ID."""
    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be a 2-D integer array")
    if (values < 0).any():
        raise ValueError("labels must be non-negative")
    unique = np.unique(values)
    return [(int(label_id), values == label_id) for label_id in unique if label_id > 0]


def _segmentation_diagnostics(labels: np.ndarray, image_shape: tuple[int, int], min_area: int, max_area_frac: float) -> tuple[float, float, float, float, float, float, float]:
    label_arr = np.asarray(labels)
    if label_arr.ndim != 2 or label_arr.shape != image_shape:
        raise ValueError("labels must be a 2-D array matching image_shape")
    if min_area < 1 or not np.isfinite(max_area_frac) or not 0 < max_area_frac <= 1:
        raise ValueError("invalid segmentation area limits")
    objects = _instance_objects(label_arr)
    if not objects:
        return 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
    areas: list[float] = []
    border_count = 0
    height, width = image_shape
    for _, mask in objects:
        ys, xs = np.nonzero(mask)
        area = float(len(xs))
        areas.append(area)
        if xs.min() == 0 or ys.min() == 0 or xs.max() == width - 1 or ys.max() == height - 1:
            border_count += 1
    areas_arr = np.asarray(areas, dtype=float)
    total_pixels = float(height * width)
    foreground_fraction = float(areas_arr.sum() / total_pixels) if total_pixels else 0.0
    median_area = float(np.median(areas_arr))
    area_cv = _safe_cv(areas_arr)
    object_count = len(objects)
    border_fraction = float(border_count / object_count)
    tiny_fraction = float((areas_arr < max(1, min_area * 2)).mean())
    merged_fraction = float((areas_arr > total_pixels * max_area_frac).mean()) if areas_arr.size else 0.0
    quality = 100.0 - min(45.0, border_fraction * 30.0) - min(30.0, tiny_fraction * 30.0) - min(25.0, merged_fraction * 25.0)
    return foreground_fraction, median_area, area_cv, border_fraction, tiny_fraction, merged_fraction, float(np.clip(quality, 0.0, 100.0))


def _build_segmentation_result(labels: np.ndarray, gray: np.ndarray, method: str, min_area: int, max_area_frac: float, error: Optional[str] = None) -> SegmentationResult:
    labels_arr = np.asarray(labels)
    gray_arr = np.asarray(gray)
    if labels_arr.ndim != 2 or gray_arr.ndim != 2 or labels_arr.shape != gray_arr.shape:
        raise ValueError("labels and gray must be matching 2-D arrays")
    if not np.issubdtype(labels_arr.dtype, np.integer):
        raise ValueError("labels must contain integer instance IDs")
    if (labels_arr < 0).any():
        raise ValueError("labels must be non-negative")
    if labels_arr.size and int(labels_arr.max()) > np.iinfo(np.int32).max:
        raise ValueError("labels contain an instance ID outside int32 range")
    positive_labels = labels_arr[labels_arr > 0]
    count = int(np.unique(positive_labels).size) if positive_labels.size else 0
    d = _segmentation_diagnostics(labels_arr, gray_arr.shape, min_area, max_area_frac)
    return SegmentationResult(count, labels_arr.astype(np.int32, copy=False), method, d[0], d[1], d[2], d[3], d[4], d[5], d[6], error)


def segment_threshold(gray: np.ndarray, min_area: int = 15, max_area_frac: float = 0.25, adaptive: bool = False) -> SegmentationResult:
    arr = np.asarray(gray)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("gray must be a non-empty 2-D image")
    if arr.dtype != np.uint8:
        raise ValueError("segment_threshold expects a 2-D uint8 working image")
    if min_area < 1 or not np.isfinite(max_area_frac) or not 0 < max_area_frac <= 1:
        raise ValueError("invalid segmentation area limits")
    blurred = cv2.GaussianBlur(arr, (3, 3), 0)
    if adaptive:
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
    else:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float((thresh > 0).mean()) > 0.5:
        thresh = cv2.bitwise_not(thresh)
    fg, bg = blurred[thresh > 0], blurred[thresh == 0]
    if fg.size == 0 or bg.size == 0 or abs(float(fg.mean()) - float(bg.mean())) < 8:
        return _build_segmentation_result(np.zeros_like(arr, dtype=np.int32), arr, "threshold", min_area, max_area_frac)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, raw_labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    max_area = arr.shape[0] * arr.shape[1] * max_area_frac
    keep = np.zeros_like(raw_labels, dtype=np.int32)
    next_id = 1
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            keep[raw_labels == label_id] = next_id
            next_id += 1
    return _build_segmentation_result(keep, arr, "threshold", min_area, max_area_frac)


class CellposeSegmenter:
    """Reusable Cellpose wrapper compatible with common Cellpose 3/4 APIs."""
    def __init__(self, model_type: str = "cpsam", gpu: bool | None = None, diameter: Optional[float] = None) -> None:
        if not _HAS_CELLPOSE:
            raise RuntimeError(f"Cellpose is not available: {_CELLPOSE_IMPORT_ERROR or 'unknown error'}. Install with: pip install cellpose")
        if not isinstance(model_type, str) or not model_type.strip():
            raise ValueError("model_type must be a non-empty string")
        if diameter is not None and (not np.isfinite(diameter) or diameter <= 0):
            raise ValueError("diameter must be a finite positive value")
        self.model_type, self.gpu, self.default_diameter, self._model = model_type, gpu, diameter, None

    @property
    def model(self):
        if self._model is not None:
            return self._model
        last_err: Optional[Exception] = None
        for kwargs0 in ({"pretrained_model": self.model_type}, {"model_type": self.model_type}, {}):
            kwargs = {**kwargs0, **({"gpu": bool(self.gpu)} if self.gpu is not None else {})}
            try:
                self._model = _cellpose_models.CellposeModel(**kwargs)
                return self._model
            except Exception as exc:
                last_err = exc
            if hasattr(_cellpose_models, "Cellpose"):
                try:
                    self._model = _cellpose_models.Cellpose(**kwargs)
                    return self._model
                except Exception as exc:
                    last_err = exc
        raise RuntimeError(f"Could not initialize Cellpose model: {last_err}")

    def segment(self, gray: np.ndarray, diameter: Optional[float] = None, min_area: int = 15, max_area_frac: float = 0.25) -> SegmentationResult:
        arr = np.asarray(gray)
        if arr.ndim != 2 or arr.size == 0 or arr.dtype != np.uint8:
            raise ValueError("Cellpose segmentation expects a non-empty 2-D uint8 working image")
        diam = diameter if diameter is not None else self.default_diameter
        if diam is not None and (not np.isfinite(diam) or diam <= 0):
            raise ValueError("diameter must be a finite positive value")
        if not isinstance(min_area, (int, np.integer)) or isinstance(min_area, bool) or min_area < 1:
            raise ValueError("min_area must be a positive integer")
        if not np.isfinite(max_area_frac) or not 0 < max_area_frac <= 1:
            raise ValueError("invalid segmentation area limits")
        eval_kwargs = {"diameter": diam} if diam is not None else {}
        masks, last_err = None, None
        for extra in ({}, {"channels": [0, 0]}):
            try:
                result = self.model.eval(arr, **eval_kwargs, **extra)
                masks = result[0] if isinstance(result, (tuple, list)) else result
                break
            except Exception as exc:
                last_err = exc
        if masks is None:
            raise RuntimeError(f"Cellpose eval failed: {last_err}")
        mask_array = np.asarray(masks)
        if mask_array.shape != arr.shape:
            raise ValueError(f"Cellpose returned masks with shape {mask_array.shape}, expected {arr.shape}")
        return _build_segmentation_result(np.asarray(mask_array, dtype=np.int32), arr, f"cellpose:{self.model_type}", min_area, max_area_frac)


def extract_object_features(gray: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    values_img = np.asarray(gray)
    if values_img.ndim != 2:
        raise ValueError(f"extract_object_features expects a 2-D grayscale image, got shape {values_img.shape}")
    if not np.issubdtype(values_img.dtype, np.number) or not np.isfinite(values_img.astype(np.float64, copy=False)).all():
        raise ValueError("gray must contain finite numeric values")
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 2 or labels_arr.shape != values_img.shape:
        raise ValueError("gray and labels must have identical 2-D shapes")
    if not np.issubdtype(labels_arr.dtype, np.integer):
        raise ValueError("labels must contain integer instance IDs")
    labels_arr = labels_arr.astype(np.int32, copy=False)
    if (labels_arr < 0).any():
        raise ValueError("labels must be non-negative")
    if labels_arr.size and int(labels_arr.max()) > np.iinfo(np.int32).max:
        raise ValueError("labels contain an instance ID outside int32 range")
    for label_id, mask in _instance_objects(labels_arr):
        area = int(mask.sum())
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0
        circularity = float((4 * np.pi * area) / (perimeter * perimeter)) if perimeter else 0.0
        ys, xs = np.nonzero(mask)
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
        object_values = values_img[mask]
        rows.append({"label": label_id, "area_px": area, "perimeter_px": perimeter, "circularity": float(np.clip(circularity, 0, 1)),
                     "bbox_x": x, "bbox_y": y, "bbox_width": w, "bbox_height": h, "aspect_ratio": float(w / h) if h else 0.0,
                     "centroid_x": float(xs.mean()), "centroid_y": float(ys.mean()),
                     "mean_intensity": float(object_values.mean()), "std_intensity": float(object_values.std()), "max_intensity": float(object_values.max())})
    return pd.DataFrame(rows)


def robust_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    median = float(values.median())
    mad = float((values - median).abs().median())
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.zeros(len(values)), index=series.index, dtype=float)
    return 0.6745 * (values - median) / mad


def adaptive_dataset_qc(df: pd.DataFrame, z_limit: float = 3.5) -> pd.DataFrame:
    if not np.isfinite(z_limit) or z_limit <= 0:
        raise ValueError("z_limit must be a finite positive value")
    result = df.copy()
    if result.empty:
        result["adaptive_score"] = pd.Series(dtype=float)
        return result
    z_columns = []
    for col in ["focus_score", "brightness_mean", "saturation_fraction", "contrast_std", "estimated_cells", "segmentation_quality"]:
        if col in result.columns:
            zcol = f"{col}_robust_z"
            result[zcol] = robust_zscore(result[col])
            z_columns.append(zcol)
    result["adaptive_score"] = result[z_columns].abs().max(axis=1).fillna(0.0) if z_columns else 0.0
    existing = result.get("flags", pd.Series("", index=result.index)).fillna("").astype(str)
    outlier = result["adaptive_score"] >= z_limit
    result.loc[outlier, "flags"] = [flag if "ADAPTIVE_OUTLIER" in flag else f"{flag}; ADAPTIVE_OUTLIER".strip("; ") for flag in existing.loc[outlier]]
    return result


def _empty_result(path: str, error: str, requested_method: str) -> ImageResult:
    exists = os.path.exists(path)
    return ImageResult(os.path.basename(path), os.path.abspath(path), 0, 0, 0, "unknown", 0,
                       os.path.getsize(path) / 1024.0 if exists else 0.0, sha256_file(path) if exists else "",
                       0.0, 0.0, 0.0, 0.0, 0.0, 0, requested_method, 0.0, 0.0, 0.0, 0.0, ["FAILED_TO_LOAD"], None, error)


def analyze_image(path: str, thresholds: Optional[QCThresholds] = None, cell_method: str = "threshold",
                  cellpose_segmenter: Optional[CellposeSegmenter] = None, adaptive_threshold: bool = False,
                  return_segmentation: bool = False) -> ImageResult | tuple[ImageResult, SegmentationResult]:
    thresholds = thresholds or QCThresholds()
    thresholds.validate()
    requested_method = cell_method.lower()
    if requested_method not in {"threshold", "cellpose"}:
        raise ValueError("cell_method must be 'threshold' or 'cellpose'")
    try:
        raw = load_image(path)
        gray = to_grayscale_uint8(raw)
    except Exception as exc:
        result = _empty_result(path, str(exc), requested_method)
        if return_segmentation:
            empty = np.zeros((1, 1), dtype=np.int32)
            return result, _build_segmentation_result(empty, np.zeros((1, 1), dtype=np.uint8), requested_method, 1, 1.0, str(exc))
        return result
    height, width = raw.shape[:2]
    channels = int(raw.shape[2]) if raw.ndim == 3 else 1
    focus = compute_focus_score(gray)
    brightness_mean, brightness_std = compute_brightness(gray)
    saturation_fraction = compute_saturation_fraction(gray)
    flags: list[str] = []
    if focus < thresholds.focus_min:
        flags.extend(["BLURRY", "LOW_FOCUS"])
    if brightness_mean < thresholds.brightness_min:
        flags.append("TOO_DARK")
    if brightness_mean > thresholds.brightness_max:
        flags.append("TOO_BRIGHT")
    if saturation_fraction > thresholds.saturation_max_fraction:
        flags.extend(["SATURATED", "HIGH_SATURATION"])
    if requested_method == "cellpose":
        try:
            seg = (cellpose_segmenter or CellposeSegmenter(gpu=False)).segment(gray, min_area=thresholds.min_cell_area, max_area_frac=thresholds.max_cell_area_frac)
        except Exception as exc:
            seg = segment_threshold(gray, min_area=thresholds.min_cell_area, max_area_frac=thresholds.max_cell_area_frac, adaptive=adaptive_threshold)
            seg.error = f"Cellpose fallback to threshold: {exc}"
            flags.append("SEGMENTATION_FALLBACK")
    else:
        seg = segment_threshold(gray, min_area=thresholds.min_cell_area, max_area_frac=thresholds.max_cell_area_frac, adaptive=adaptive_threshold)
    if seg.count < thresholds.cell_count_low:
        flags.extend(["FEW_OR_NO_CELLS", "LOW_CELL_COUNT"])
    if thresholds.cell_count_high is not None and seg.count > thresholds.cell_count_high:
        flags.extend(["TOO_MANY_CELLS", "HIGH_CELL_COUNT"])
    if seg.quality_score < 60:
        flags.append("LOW_SEGMENTATION_QUALITY")
    if seg.error:
        flags.append("SEGMENTATION_WARNING")
    result = ImageResult(os.path.basename(path), os.path.abspath(path), int(width), int(height), channels, str(raw.dtype), int(raw.ndim),
                         round(os.path.getsize(path) / 1024.0, 2), sha256_file(path), round(focus, 4), round(brightness_mean, 4),
                         round(brightness_std, 4), round(saturation_fraction, 6), round(brightness_std, 4), int(seg.count), seg.method,
                         round(seg.quality_score, 3), round(seg.median_area, 3), round(seg.area_cv, 5), round(seg.border_fraction, 5),
                         list(dict.fromkeys(flags)), None, seg.error)
    return (result, seg) if return_segmentation else result


def find_images(folder: str) -> list[str]:
    """Recursively discover supported files, case-insensitively."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    return sorted(str(path) for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def analyze_paths(paths: Sequence[str], thresholds: Optional[QCThresholds] = None, cell_method: str = "threshold",
                  progress_callback: Optional[Callable[[int, int, str], None]] = None, adaptive_qc: bool = True,
                  adaptive_threshold: bool = False) -> pd.DataFrame:
    """Analyze explicit paths without deduplicating distinct files."""
    thresholds = thresholds or QCThresholds()
    thresholds.validate()
    normalized = [os.fspath(path) for path in paths]
    segmenter = CellposeSegmenter(gpu=False) if cell_method.lower() == "cellpose" and _HAS_CELLPOSE else None
    rows = []
    total = len(normalized)
    for i, path in enumerate(normalized, start=1):
        result = analyze_image(path, thresholds=thresholds, cell_method=cell_method, cellpose_segmenter=segmenter, adaptive_threshold=adaptive_threshold)
        if isinstance(result, tuple):
            result = result[0]
        rows.append(result.to_row())
        if progress_callback:
            progress_callback(i, total, os.path.basename(path))
    df = pd.DataFrame(rows)
    return adaptive_dataset_qc(df) if adaptive_qc else df


def analyze_folder(folder: str, thresholds: Optional[QCThresholds] = None, cell_method: str = "threshold",
                   progress_callback: Optional[Callable[[int, int, str], None]] = None, adaptive_qc: bool = True,
                   adaptive_threshold: bool = False) -> pd.DataFrame:
    return analyze_paths(find_images(folder), thresholds=thresholds, cell_method=cell_method,
                         progress_callback=progress_callback, adaptive_qc=adaptive_qc, adaptive_threshold=adaptive_threshold)


def export_csv(df: pd.DataFrame, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def export_json(df: pd.DataFrame, out_path: str, metadata: Optional[dict[str, Any]] = None) -> str:
    payload = {"pipeline_version": PIPELINE_VERSION, "metadata": metadata or {}, "records": df.replace({np.nan: None}).to_dict(orient="records")}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_path


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="opticell", description="OptiCell quantitative microscopy QC and cell-analysis pipeline")
    parser.add_argument("input", nargs="?", help="Image file or directory")
    parser.add_argument("--paths", nargs="*", help="Image files or directories (compatibility alias)")
    parser.add_argument("-o", "--output", "--out", default="qc_summary.csv")
    parser.add_argument("--json", dest="json_output")
    parser.add_argument("--cell-method", "--method", choices=["threshold", "cellpose"], default="threshold")
    parser.add_argument("--adaptive-threshold", action="store_true")
    parser.add_argument("--no-adaptive-qc", action="store_true")
    parser.add_argument("--focus-min", type=float, default=100.0)
    parser.add_argument("--brightness-min", type=float, default=25.0)
    parser.add_argument("--brightness-max", type=float, default=230.0)
    parser.add_argument("--min-cell-area", type=int, default=15)
    parser.add_argument("--max-cell-area-frac", type=float, default=0.25)
    parser.add_argument("--cell-count-low", type=int, default=1)
    parser.add_argument("--cell-count-high", type=int, default=None)
    args = parser.parse_args(argv)
    inputs = list(args.paths or ([] if args.input is None else [args.input]))
    if not inputs:
        parser.error("provide an input path or --paths")
    thresholds = QCThresholds(focus_min=args.focus_min, brightness_min=args.brightness_min, brightness_max=args.brightness_max,
                              min_cell_area=args.min_cell_area, max_cell_area_frac=args.max_cell_area_frac,
                              cell_count_low=args.cell_count_low, cell_count_high=args.cell_count_high)
    expanded = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            expanded.extend(find_images(str(p)))
        elif p.is_file():
            expanded.append(str(p))
        else:
            parser.error(f"Input path does not exist: {item}")
    files = sorted(dict.fromkeys(expanded))
    if not files:
        print("No images found")
        return 1
    def progress(done: int, total: int, name: str) -> None:
        print(f"[{done}/{total}] {name}")
    df = analyze_paths(files, thresholds=thresholds, cell_method=args.cell_method, progress_callback=progress,
                       adaptive_qc=not args.no_adaptive_qc, adaptive_threshold=args.adaptive_threshold)
    export_csv(df, args.output)
    if args.json_output:
        export_json(df, args.json_output, metadata={"pipeline_version": PIPELINE_VERSION, "cell_method_requested": args.cell_method,
                                                     "adaptive_qc": not args.no_adaptive_qc, "input": [os.path.abspath(p) for p in inputs],
                                                     "thresholds": asdict(thresholds)})
    flagged = int((df["flags"].fillna("") != "").sum()) if not df.empty else 0
    failed = int(df["error"].notna().sum()) if not df.empty else 0
    print(f"OptiCell {PIPELINE_VERSION}: {len(df)} images analyzed")
    print(f"Flagged: {flagged} | Failed: {failed}")
    print(f"CSV: {args.output}")
    if args.json_output:
        print(f"JSON: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
