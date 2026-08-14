"""
qc_pipeline.py
================
Core analysis engine for the microscopy QC tool.

Responsibilities
-----------------
- Load microscopy images (.png, .jpg/.jpeg, .tif/.tiff, incl. multi-page/16-bit TIFFs)
- Compute per-image QC metrics:
    * Focus / sharpness score  (variance of Laplacian)
    * Brightness score         (mean intensity, 0-255 scale)
    * Estimated cell count     (Otsu threshold + connected components,
                                 with an optional Cellpose backend)
- Flag images that fall outside expected ranges (blurry, too dark/bright,
  suspiciously low/high cell counts)
- Batch-process a whole folder into a pandas DataFrame
- Export a summary CSV

This module has no GUI dependencies — it can be used standalone from a
script/notebook, from the CLI, or imported by app.py (the Streamlit GUI).
"""

from __future__ import annotations

import os
import glob
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import cv2
import pandas as pd

try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False

# Cellpose is a heavy optional dependency (torch-based). The tool works
# perfectly well without it, using classical thresholding instead.
try:
    from cellpose import models as _cellpose_models
    _HAS_CELLPOSE = True
except ImportError:
    _HAS_CELLPOSE = False


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class QCThresholds:
    """Cutoffs used to flag suspicious images. Tune these to your dataset."""
    focus_min: float = 100.0          # variance-of-Laplacian below this => blurry
    brightness_min: float = 25.0      # mean intensity below this => too dark
    brightness_max: float = 230.0     # mean intensity above this => too bright / saturated
    min_cell_area: int = 15           # pixels; connected components smaller than this are noise
    max_cell_area_frac: float = 0.25  # component larger than this fraction of image => merged blob, not a cell
    cell_count_low: int = 1           # fewer cells than this => flag "few/no cells detected"


@dataclass
class ImageResult:
    """QC result for a single image."""
    filename: str
    path: str
    width: int
    height: int
    channels: int
    dtype: str
    file_size_kb: float
    focus_score: float
    brightness_mean: float
    brightness_std: float
    estimated_cells: int
    cell_method: str
    flags: list = field(default_factory=list)
    error: Optional[str] = None

    def to_row(self) -> dict:
        d = dataclasses.asdict(self)
        d["flags"] = "; ".join(self.flags) if self.flags else ""
        return d


# --------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk as a numpy array, handling multi-page and
    high-bit-depth TIFFs gracefully. Returns the array in its native
    dtype/shape (H, W) or (H, W, C).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".tif", ".tiff") and _HAS_TIFFFILE:
        arr = tifffile.imread(path)
        # Multi-page/Z-stack TIFF -> collapse to a single 2D image via max projection
        if arr.ndim == 3 and arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 64:
            # heuristic: first axis is likely pages/z-slices, not a small channel count like RGB(3)/RGBA(4)
            if arr.shape[0] not in (3, 4):
                arr = arr.max(axis=0)
        return arr

    # Fall back to OpenCV for everything else (also handles standard TIFFs
    # if tifffile isn't installed)
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read image: {path}")
    # OpenCV loads color images as BGR -> convert to RGB for consistency
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    return arr


def to_grayscale_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize any loaded image (any dtype/shape) to a 2D uint8 grayscale array."""
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            gray = cv2.cvtColor(arr[:, :, :3].astype(np.uint8) if arr.dtype == np.uint8
                                 else _rescale_to_uint8(arr[:, :, :3]), cv2.COLOR_RGB2GRAY)
            return gray
        else:
            arr = arr[:, :, 0]

    if arr.dtype == np.uint8:
        return arr

    return _rescale_to_uint8(arr)


def _rescale_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Rescale arbitrary-range/dtype image data (e.g. 12/16-bit) to 0-255 uint8."""
    arr = arr.astype(np.float64)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
    if hi <= lo:
        return np.zeros(arr.shape[:2] if arr.ndim == 3 else arr.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255.0
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr.astype(np.uint8)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_focus_score(gray: np.ndarray) -> float:
    """
    Sharpness/focus metric: variance of the Laplacian.
    Higher = sharper. Blurry images have low edge energy -> low variance.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(gray: np.ndarray) -> tuple[float, float]:
    """Returns (mean intensity, std intensity) on a 0-255 scale."""
    return float(gray.mean()), float(gray.std())


def estimate_cell_count_threshold(
    gray: np.ndarray,
    min_area: int = 15,
    max_area_frac: float = 0.25,
) -> tuple[int, np.ndarray]:
    """
    Classical cell-count estimate:
      1. Denoise slightly (median blur)
      2. Otsu threshold (auto picks foreground vs background — works whether
         cells are bright-on-dark or dark-on-bright, we pick the smaller-area side)
      3. Morphological opening to remove speckle noise
      4. Connected-component labeling, filtered by plausible area

    Returns (cell_count, labeled_image) where labeled_image is an int32
    array useful for visualization/overlay.
    """
    blurred = cv2.medianBlur(gray, 3)
    thresh_val, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Decide polarity: foreground (cells) is usually the minority class.
    # If thresholded "foreground" covers >50% of the image, invert.
    if (thresh > 0).mean() > 0.5:
        thresh = cv2.bitwise_not(thresh)

    # Guard against flat/empty fields: on pure sensor noise, Otsu still
    # finds *a* split, but foreground and background pixel intensities are
    # nearly identical. Require a real intensity gap before trusting it.
    fg_vals = blurred[thresh > 0]
    bg_vals = blurred[thresh == 0]
    if fg_vals.size == 0 or bg_vals.size == 0 or (fg_vals.mean() - bg_vals.mean()) < 12:
        empty_labels = np.zeros_like(blurred, dtype=np.int32)
        return 0, empty_labels

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    total_pixels = gray.shape[0] * gray.shape[1]
    max_area = total_pixels * max_area_frac

    count = 0
    keep_mask = np.zeros_like(labels, dtype=bool)
    for label_id in range(1, num_labels):  # skip background label 0
        area = stats[label_id, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            count += 1
            keep_mask |= (labels == label_id)

    filtered_labels = np.where(keep_mask, labels, 0)
    return count, filtered_labels


def estimate_cell_count_cellpose(gray: np.ndarray, diameter: Optional[float] = None) -> tuple[int, np.ndarray]:
    """
    Optional higher-accuracy cell count using Cellpose ('cyto' model).
    Only used if the `cellpose` package is installed; falls back to the
    threshold method otherwise (see analyze_image).
    """
    if not _HAS_CELLPOSE:
        raise RuntimeError("cellpose is not installed")
    model = _cellpose_models.Cellpose(model_type="cyto")
    masks, _, _, _ = model.eval(gray, diameter=diameter, channels=[0, 0])
    count = int(masks.max())
    return count, masks


# --------------------------------------------------------------------------
# Single-image analysis
# --------------------------------------------------------------------------

def analyze_image(
    path: str,
    thresholds: QCThresholds = QCThresholds(),
    cell_method: str = "threshold",  # "threshold" or "cellpose"
) -> ImageResult:
    """Run the full QC pipeline on one image file and return an ImageResult."""
    filename = os.path.basename(path)
    file_size_kb = os.path.getsize(path) / 1024.0

    try:
        raw = load_image(path)
    except Exception as e:
        return ImageResult(
            filename=filename, path=path, width=0, height=0, channels=0,
            dtype="unknown", file_size_kb=round(file_size_kb, 1),
            focus_score=0.0, brightness_mean=0.0, brightness_std=0.0,
            estimated_cells=0, cell_method=cell_method,
            flags=["FAILED_TO_LOAD"], error=str(e),
        )

    height, width = raw.shape[0], raw.shape[1]
    channels = raw.shape[2] if raw.ndim == 3 else 1
    dtype = str(raw.dtype)

    gray = to_grayscale_uint8(raw)

    focus = compute_focus_score(gray)
    brightness_mean, brightness_std = compute_brightness(gray)

    method_used = cell_method
    try:
        if cell_method == "cellpose" and _HAS_CELLPOSE:
            n_cells, _ = estimate_cell_count_cellpose(gray)
        else:
            method_used = "threshold"
            n_cells, _ = estimate_cell_count_threshold(
                gray, min_area=thresholds.min_cell_area, max_area_frac=thresholds.max_cell_area_frac
            )
    except Exception:
        method_used = "threshold"
        n_cells, _ = estimate_cell_count_threshold(
            gray, min_area=thresholds.min_cell_area, max_area_frac=thresholds.max_cell_area_frac
        )

    flags = []
    if focus < thresholds.focus_min:
        flags.append("BLURRY")
    if brightness_mean < thresholds.brightness_min:
        flags.append("TOO_DARK")
    if brightness_mean > thresholds.brightness_max:
        flags.append("TOO_BRIGHT")
    if n_cells < thresholds.cell_count_low:
        flags.append("FEW_OR_NO_CELLS")

    return ImageResult(
        filename=filename, path=path, width=width, height=height, channels=channels,
        dtype=dtype, file_size_kb=round(file_size_kb, 1),
        focus_score=round(focus, 2), brightness_mean=round(brightness_mean, 2),
        brightness_std=round(brightness_std, 2), estimated_cells=n_cells,
        cell_method=method_used, flags=flags,
    )


# --------------------------------------------------------------------------
# Batch / folder processing
# --------------------------------------------------------------------------

def find_images(folder: str) -> list:
    """Recursively find all supported image files under a folder."""
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, "**", f"*{ext}"), recursive=True))
        paths.extend(glob.glob(os.path.join(folder, "**", f"*{ext.upper()}"), recursive=True))
    return sorted(set(paths))


def analyze_folder(
    folder: str,
    thresholds: QCThresholds = QCThresholds(),
    cell_method: str = "threshold",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """
    Analyze every supported image in `folder` (recursively) and return a
    tidy pandas DataFrame, one row per image.

    progress_callback(done, total, current_filename) is called after each
    image if provided (used by the Streamlit progress bar).
    """
    paths = find_images(folder)
    rows = []
    for i, path in enumerate(paths, start=1):
        result = analyze_image(path, thresholds=thresholds, cell_method=cell_method)
        rows.append(result.to_row())
        if progress_callback:
            progress_callback(i, len(paths), os.path.basename(path))
    return pd.DataFrame(rows)


def analyze_paths(
    paths: list,
    thresholds: QCThresholds = QCThresholds(),
    cell_method: str = "threshold",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Same as analyze_folder but takes an explicit list of file paths (used
    when images arrive via a file-uploader rather than a folder path)."""
    rows = []
    for i, path in enumerate(paths, start=1):
        result = analyze_image(path, thresholds=thresholds, cell_method=cell_method)
        rows.append(result.to_row())
        if progress_callback:
            progress_callback(i, len(paths), os.path.basename(path))
    return pd.DataFrame(rows)


def export_csv(df: pd.DataFrame, out_path: str) -> str:
    df.to_csv(out_path, index=False)
    return out_path


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Microscopy image QC pipeline")
    parser.add_argument("folder", help="Folder of microscopy images to analyze")
    parser.add_argument("-o", "--output", default="qc_summary.csv", help="Output CSV path")
    parser.add_argument("--cell-method", default="threshold", choices=["threshold", "cellpose"])
    parser.add_argument("--focus-min", type=float, default=100.0)
    parser.add_argument("--brightness-min", type=float, default=25.0)
    parser.add_argument("--brightness-max", type=float, default=230.0)
    args = parser.parse_args()

    thresholds = QCThresholds(
        focus_min=args.focus_min,
        brightness_min=args.brightness_min,
        brightness_max=args.brightness_max,
    )

    def _progress(done, total, name):
        print(f"[{done}/{total}] {name}")

    df = analyze_folder(args.folder, thresholds=thresholds, cell_method=args.cell_method,
                         progress_callback=_progress)
    export_csv(df, args.output)
    n_flagged = (df["flags"] != "").sum()
    print(f"\nDone. {len(df)} images analyzed, {n_flagged} flagged for review.")
    print(f"Summary written to {args.output}")


if __name__ == "__main__":
    _main()
