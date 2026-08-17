"""OptiCell microscopy analysis engine.

This module intentionally has no GUI dependencies. It provides a reusable
Python API and CLI for microscopy dataset QC and quantitative cell analysis.

Key capabilities
----------------
* Robust loading of common image formats, including multi-page TIFFs.
* Exposure, saturation, contrast and focus metrics.
* Classical segmentation with explicit diagnostics.
* Optional persistent Cellpose backend when installed.
* Per-object morphology features.
* Dataset-level adaptive QC using robust MAD-based scores.
* Deterministic, provenance-friendly CSV/JSON export.
* Recursive batch processing with progress callbacks.
"""

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
    """Absolute limits used alongside dataset-level adaptive QC."""

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
    """Conservatively collapse non-RGB TIFF dimensions to a 2-D projection."""
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
    """Load a supported image as a NumPy array in RGB order where applicable."""
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
    """Convert arbitrary numeric image data to uint8 using robust percentiles."""
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
    """Convert grayscale/RGB/RGBA/high-bit-depth input to 2-D uint8."""
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


def _segmentation_diagnostics(
    labels: np.ndarray,
    image_shape: tuple[int, int],
    min_area: int,
    max_area_frac: float,
) -> tuple[float, float, float, float, float, float, float]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        (labels > 0).astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0

    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    total_pixels = float(image_shape[0] * image_shape[1])
    foreground_fraction = float(areas.sum() / total_pixels) if total_pixels else 0.0
    median_area = float(np.median(areas))
    area_cv = _safe_cv(areas)

    border_count = 0
    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
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


def _build_segmentation_result(
    labels: np.ndarray,
    gray: np.ndarray,
    method: str,
    min_area: int,
    max_area_frac: float,
    error: Optional[str] = None,
) -> SegmentationResult:
    count = int(labels.max()) if labels.size else 0
    (
        foreground_fraction,
        median_area,
        area_cv,
        border_fraction,
        tiny_fraction,
        merged_fraction,
        quality,
    ) = _segmentation_diagnostics(labels, gray.shape, min_area, max_area_frac)
    return SegmentationResult(
        count=count,
        labels=labels.astype(np.int32, copy=False),
        method=method,
        foreground_fraction=foreground_fraction,
        median_area=median_area,
        area_cv=area_cv,
        border_fraction=border_fraction,
        tiny_object_fraction=tiny_fraction,
        merged_object_fraction=merged_fraction,
        quality_score=quality,
        error=error,
    )


def segment_threshold(
    gray: np.ndarray,
    min_area: int = 15,
    max_area_frac: float = 0.25,
    adaptive: bool = False,
) -> SegmentationResult:
    """Threshold + morphology + connected components, with diagnostics."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    if adaptive:
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3
        )
    else:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if float((thresh > 0).mean()) > 0.5:
        thresh = cv2.bitwise_not(thresh)
    fg = blurred[thresh > 0]
    bg = blurred[thresh == 0]
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
    """Reusable Cellpose model wrapper; model initialization happens once."""

    def __init__(self, model_type: str = "cyto3") -> None:
        if not _HAS_CELLPOSE:
            raise RuntimeError("Cellpose is not installed. Install the optional cellpose dependency.")
        self.model_type = model_type
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                self._model = _cellpose_models.CellposeModel(model_type=self.model_type)
            except AttributeError:
                self._model = _cellpose_models.Cellpose(model_type=self.model_type)
        return self._model

    def segment(
        self,
        gray: np.ndarray,
        diameter: Optional[float] = None,
        min_area: int = 15,
        max_area_frac: float = 0.25,
    ) -> SegmentationResult:
        result = self.model.eval(gray, diameter=diameter, channels=[0, 0])
        masks = result[0]
        labels = np.asarray(masks, dtype=np.int32)
        return _build_segmentation_result(labels, gray, f"cellpose:{self.model_type}", min_area, max_area_frac)


def extract_object_features(gray: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Extract morphology/intensity measurements, one row per object."""
    rows: list[dict[str, Any]] = []
    labels = labels.astype(np.int32, copy=False)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        (labels > 0).astype(np.uint8), connectivity=8
    )
    for label_id in range(1, num_labels):
        mask = labels == label_id
        area = int(mask.sum())
        if area <= 0:
            continue
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0
        circularity = float((4 * np.pi * area) / (perimeter * perimeter)) if perimeter else 0.0
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        values = gray[mask]
        rows.append(
            {
                "label": label_id,
                "area_px": area,
                "perimeter_px": perimeter,
                "circularity": float(np.clip(circularity, 0, 1)),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": w,
                "bbox_height": h,
                "aspect_ratio": float(w / h) if h else 0.0,
                "centroid_x": float(centroids[label_id][0]),
                "centroid_y": float(centroids[label_id][1]),
                "mean_intensity": float(values.mean()),
                "std_intensity": float(values.std()),
                "max_intensity": int(values.max()),
            }
        )
    return pd.DataFrame(rows)


def robust_zscore(series: pd.Series) -> pd.Series:
    """Median/MAD z-score, robust to genuine biological outliers."""
    values = pd.to_numeric(series, errors="coerce")
    median = float(values.median())
    mad = float((values - median).abs().median())
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.zeros(len(values)), index=series.index, dtype=float)
    return 0.6745 * (values - median) / mad


def adaptive_dataset_qc(df: pd.DataFrame, z_limit: float = 3.5) -> pd.DataFrame:
    """Append robust outlier scores without erasing existing QC flags."""
    result = df.copy()
    if result.empty:
        result["adaptive_score"] = pd.Series(dtype=float)
        return result

    metric_cols = [
        "focus_score",
        "brightness_mean",
        "saturation_fraction",
        "contrast_std",
        "estimated_cells",
        "segmentation_quality",
    ]
    z_columns: list[str] = []
    for col in metric_cols:
        if col in result.columns:
            zcol = f"{col}_robust_z"
            result[zcol] = robust_zscore(result[col])
            z_columns.append(zcol)

    result["adaptive_score"] = result[z_columns].abs().max(axis=1).fillna(0.0) if z_columns else 0.0
    existing = result.get("flags", pd.Series("", index=result.index)).fillna("").astype(str)
    is_outlier = result["adaptive_score"] >= z_limit
    result.loc[is_outlier, "flags"] = [
        f"{flag}; ADAPTIVE_OUTLIER".strip("; ") if "ADAPTIVE_OUTLIER" not in flag else flag
        for flag in existing.loc[is_outlier]
    ]
    return result


def _empty_result(path: str, error: str, requested_method: str) -> ImageResult:
    exists = os.path.exists(path)
    file_size = os.path.getsize(path) / 1024.0 if exists else 0.0
    digest = sha256_file(path) if exists else ""
    return ImageResult(
        filename=os.path.basename(path),
        path=os.path.abspath(path),
        width=0,
        height=0,
        channels=0,
        dtype="unknown",
        ndim=0,
        file_size_kb=round(file_size, 2),
        sha256=digest,
        focus_score=0.0,
        brightness_mean=0.0,
        brightness_std=0.0,
        saturation_fraction=0.0,
        contrast_std=0.0,
        estimated_cells=0,
        cell_method=requested_method,
        segmentation_quality=0.0,
        median_cell_area=0.0,
        cell_area_cv=0.0,
        border_object_fraction=0.0,
        flags=["FAILED_TO_LOAD"],
        error=error,
    )


def analyze_image(
    path: str,
    thresholds: Optional[QCThresholds] = None,
    cell_method: str = "threshold",
    cellpose_segmenter: Optional[CellposeSegmenter] = None,
    adaptive_threshold: bool = False,
    return_segmentation: bool = False,
) -> ImageResult | tuple[ImageResult, SegmentationResult]:
    """Analyze one image; request segmentation masks when needed for downstream work."""
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
            dummy = _build_segmentation_result(empty, np.zeros((1, 1), dtype=np.uint8), requested_method, 1, 1.0, str(exc))
            return result, dummy
        return result

    height, width = raw.shape[:2]
    channels = raw.shape[2] if raw.ndim == 3 else 1
    focus = compute_focus_score(gray)
    brightness_mean, brightness_std = compute_brightness(gray)
    saturation_fraction = compute_saturation_fraction(gray)

    flags: list[str] = []
    if focus < thresholds.focus_min:
        flags.append("BLURRY")
    if brightness_mean < thresholds.brightness_min:
        flags.append("TOO_DARK")
    if brightness_mean > thresholds.brightness_max:
        flags.append("TOO_BRIGHT")
    if saturation_fraction > thresholds.saturation_max_fraction:
        flags.append("SATURATED")

    seg: SegmentationResult
    if requested_method == "cellpose":
        try:
            backend = cellpose_segmenter or CellposeSegmenter()
            seg = backend.segment(gray, min_area=thresholds.min_cell_area, max_area_frac=thresholds.max_cell_area_frac)
        except Exception as exc:
            seg = segment_threshold(
                gray,
                min_area=thresholds.min_cell_area,
                max_area_frac=thresholds.max_cell_area_frac,
                adaptive=adaptive_threshold,
            )
            seg.error = f"Cellpose fallback to threshold: {exc}"
            flags.append("SEGMENTATION_FALLBACK")
    else:
        seg = segment_threshold(
            gray,
            min_area=thresholds.min_cell_area,
            max_area_frac=thresholds.max_cell_area_frac,
            adaptive=adaptive_threshold,
        )

    if seg.count < thresholds.cell_count_low:
        flags.append("FEW_OR_NO_CELLS")
    if thresholds.cell_count_high is not None and seg.count > thresholds.cell_count_high:
        flags.append("TOO_MANY_CELLS")
    if seg.quality_score < 60:
        flags.append("LOW_SEGMENTATION_QUALITY")
    if seg.error:
        flags.append("SEGMENTATION_WARNING")

    result = ImageResult(
        filename=os.path.basename(path),
        path=os.path.abspath(path),
        width=int(width),
        height=int(height),
        channels=int(channels),
        dtype=str(raw.dtype),
        ndim=int(raw.ndim),
        file_size_kb=round(os.path.getsize(path) / 1024.0, 2),
        sha256=sha256_file(path),
        focus_score=round(focus, 4),
        brightness_mean=round(brightness_mean, 4),
        brightness_std=round(brightness_std, 4),
        saturation_fraction=round(saturation_fraction, 6),
        contrast_std=round(brightness_std, 4),
        estimated_cells=int(seg.count),
        cell_method=seg.method,
        segmentation_quality=round(seg.quality_score, 3),
        median_cell_area=round(seg.median_area, 3),
        cell_area_cv=round(seg.area_cv, 5),
        border_object_fraction=round(seg.border_fraction, 5),
        flags=flags,
        error=seg.error,
    )
    return (result, seg) if return_segmentation else result


def find_images(folder: str) -> list[str]:
    """Recursively discover supported files, case-insensitively and once."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    return sorted(str(path) for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def analyze_paths(
    paths: Sequence[str],
    thresholds: Optional[QCThresholds] = None,
    cell_method: str = "threshold",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    adaptive_qc: bool = True,
    adaptive_threshold: bool = False,
) -> pd.DataFrame:
    """Analyze explicit paths and optionally append dataset-level adaptive QC."""
    thresholds = thresholds or QCThresholds()
    thresholds.validate()
    normalized = [os.fspath(p) for p in paths]
    rows: list[dict[str, Any]] = []
    segmenter = CellposeSegmenter() if cell_method == "cellpose" and _HAS_CELLPOSE else None

    total = len(normalized)
    for i, path in enumerate(normalized, start=1):
        result = analyze_image(
            path,
            thresholds=thresholds,
            cell_method=cell_method,
            cellpose_segmenter=segmenter,
            adaptive_threshold=adaptive_threshold,
        )
        if isinstance(result, tuple):
            result = result[0]
        rows.append(result.to_row())
        if progress_callback:
            progress_callback(i, total, os.path.basename(path))

    df = pd.DataFrame(rows)
    return adaptive_dataset_qc(df) if adaptive_qc else df


def analyze_folder(
    folder: str,
    thresholds: Optional[QCThresholds] = None,
    cell_method: str = "threshold",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    adaptive_qc: bool = True,
    adaptive_threshold: bool = False,
) -> pd.DataFrame:
    return analyze_paths(
        find_images(folder),
        thresholds=thresholds,
        cell_method=cell_method,
        progress_callback=progress_callback,
        adaptive_qc=adaptive_qc,
        adaptive_threshold=adaptive_threshold,
    )


def export_csv(df: pd.DataFrame, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def export_json(df: pd.DataFrame, out_path: str, metadata: Optional[dict[str, Any]] = None) -> str:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "metadata": metadata or {},
        "records": df.replace({np.nan: None}).to_dict(orient="records"),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_path


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="opticell", description="OptiCell quantitative microscopy QC and cell-analysis pipeline"
    )
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("-o", "--output", default="qc_summary.csv")
    parser.add_argument("--json", dest="json_output")
    parser.add_argument("--cell-method", choices=["threshold", "cellpose"], default="threshold")
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

    thresholds = QCThresholds(
        focus_min=args.focus_min,
        brightness_min=args.brightness_min,
        brightness_max=args.brightness_max,
        min_cell_area=args.min_cell_area,
        max_cell_area_frac=args.max_cell_area_frac,
        cell_count_low=args.cell_count_low,
        cell_count_high=args.cell_count_high,
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        paths = find_images(str(input_path))
    elif input_path.is_file():
        paths = [str(input_path)]
    else:
        parser.error(f"Input path does not exist: {args.input}")
        return 2

    def progress(done: int, total: int, name: str) -> None:
        print(f"[{done}/{total}] {name}")

    df = analyze_paths(
        paths,
        thresholds=thresholds,
        cell_method=args.cell_method,
        progress_callback=progress,
        adaptive_qc=not args.no_adaptive_qc,
        adaptive_threshold=args.adaptive_threshold,
    )
    export_csv(df, args.output)
    if args.json_output:
        export_json(
            df,
            args.json_output,
            metadata={
                "pipeline_version": PIPELINE_VERSION,
                "cell_method_requested": args.cell_method,
                "adaptive_qc": not args.no_adaptive_qc,
                "input": os.path.abspath(args.input),
                "thresholds": asdict(thresholds),
            },
        )

    flagged = int((df["flags"].fillna("") != "").sum()) if not df.empty else 0
    failed = int(df["error"].notna().sum()) if not df.empty else 0
    print(f"\nOptiCell {PIPELINE_VERSION}: {len(df)} images analyzed")
    print(f"Flagged: {flagged} | Failed: {failed}")
    print(f"CSV: {args.output}")
    if args.json_output:
        print(f"JSON: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
