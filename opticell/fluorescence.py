"""Opt-in classical segmentation for bright, heterogeneous fluorescence nuclei.

Logarithmic Otsu reduces the dominance of very bright objects; enclosed holes
are filled before distance-watershed splitting. This is a morphology heuristic,
not a trained model, and is unsuitable for phase contrast or dark objects.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from qc_pipeline import SegmentationResult, _build_segmentation_result


def segment_fluorescence(
    gray: np.ndarray, min_area: int = 15, max_area_frac: float = 0.25
) -> SegmentationResult:
    """Segment bright nuclei; scale the watershed neighbourhood from image areas.

    Parameters are image-derived, never read from reference annotations. Import
    the optional scikit-image dependency only when this backend is requested.
    """
    try:
        from skimage.segmentation import watershed
    except ImportError as exc:
        raise RuntimeError("Fluorescence backend requires: pip install 'opticell[fluorescence]'") from exc

    arr = np.asarray(gray)
    if arr.ndim != 2 or arr.size == 0 or arr.dtype != np.uint8:
        raise ValueError("segment_fluorescence expects a non-empty 2-D uint8 working image")
    if min_area < 1 or not np.isfinite(max_area_frac) or not 0 < max_area_frac <= 1:
        raise ValueError("invalid segmentation area limits")
    if arr.min() == arr.max():
        return _build_segmentation_result(
            np.zeros(arr.shape, np.int32), arr, "fluorescence:log_otsu_watershed", min_area, max_area_frac
        )
    smooth = cv2.GaussianBlur(arr.astype(np.float32), (5, 5), 1)
    logged = np.log1p(smooth)
    working = np.uint8(logged / logged.max() * 255)
    _, foreground = cv2.threshold(working, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    foreground = ndi.binary_fill_holes(foreground)
    components, _ = ndi.label(foreground)
    areas = np.bincount(components.ravel())
    keep = areas >= min_area
    keep[0] = False
    foreground = keep[components]
    distance = ndi.distance_transform_edt(foreground)
    admitted_areas = areas[1:][areas[1:] >= min_area]
    radius = max(2, int(np.sqrt(np.median(admitted_areas) / np.pi) * 0.5)) if admitted_areas.size else 2
    peaks = (distance == ndi.maximum_filter(distance, size=2 * radius + 1)) & (distance > 0)
    markers, _ = ndi.label(peaks)
    labels = watershed(-distance, markers, mask=foreground)
    # Apply area limits again after splitting and assign contiguous instance IDs.
    sizes = np.bincount(labels.ravel())
    keep = (sizes >= min_area) & (sizes <= arr.size * max_area_frac)
    keep[0] = False
    lookup = np.where(keep, np.cumsum(keep), 0).astype(np.int32)
    return _build_segmentation_result(
        lookup[labels], arr, "fluorescence:log_otsu_watershed", min_area, max_area_frac
    )


class FluorescenceSegmenter:
    name = "fluorescence"

    def __init__(self, min_area: int = 15, max_area_frac: float = 0.25) -> None:
        self.min_area = min_area
        self.max_area_frac = max_area_frac

    def segment(self, image: np.ndarray, **kwargs) -> SegmentationResult:
        return segment_fluorescence(
            image,
            min_area=kwargs.get("min_area", self.min_area),
            max_area_frac=kwargs.get("max_area_frac", self.max_area_frac),
        )
