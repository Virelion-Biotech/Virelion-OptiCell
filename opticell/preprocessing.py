"""Explicit, provenance-friendly microscopy preprocessing operations."""
from __future__ import annotations

import cv2
import numpy as np


def flat_field_correct(image: np.ndarray, reference: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Correct multiplicative illumination using a reference field."""
    source = np.asarray(image)
    reference_arr = np.asarray(reference)
    if source.shape != reference_arr.shape:
        raise ValueError("image and reference must have identical shapes")
    if not np.issubdtype(source.dtype, np.number) or not np.issubdtype(reference_arr.dtype, np.number):
        raise TypeError("image and reference must contain numeric values")
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("eps must be a finite positive value")
    if not np.isfinite(source.astype(np.float64, copy=False)).all():
        raise ValueError("image must contain only finite values")
    if not np.isfinite(reference_arr.astype(np.float64, copy=False)).all() or np.any(reference_arr <= 0):
        raise ValueError("reference must contain only finite, strictly positive values")
    image_f32 = source.astype(np.float32, copy=False)
    reference_f32 = reference_arr.astype(np.float32, copy=False)
    scale = float(reference_f32.mean())
    corrected = image_f32 * scale / np.maximum(reference_f32, eps)
    if np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        corrected = np.clip(corrected, info.min, info.max).astype(source.dtype)
    elif np.issubdtype(source.dtype, np.floating):
        corrected = corrected.astype(source.dtype, copy=False)
    return corrected


def estimate_background(gray: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Estimate smooth illumination/background without altering the source."""
    image = np.asarray(gray)
    if image.ndim != 2 or sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("gray must be 2-D and sigma must be a finite positive value")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("gray must contain numeric values")
    work = image.astype(np.float32, copy=False)
    if not np.isfinite(work).all():
        raise ValueError("gray must contain only finite values")
    return cv2.GaussianBlur(work, (0, 0), sigmaX=sigma)


def subtract_background(gray: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Subtract estimated background while preserving the input numeric dtype when possible."""
    image = np.asarray(gray)
    background = estimate_background(image, sigma=sigma)
    corrected = np.clip(image.astype(np.float32) - background, 0, None)
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        return np.clip(corrected, info.min, info.max).astype(image.dtype)
    if np.issubdtype(image.dtype, np.floating):
        return corrected.astype(image.dtype, copy=False)
    raise TypeError("gray must contain numeric values")


def detect_hot_pixels(gray: np.ndarray, z_threshold: float = 6.0) -> np.ndarray:
    """Return a boolean mask of unusually bright isolated pixels."""
    image = np.asarray(gray, dtype=np.float32)
    if image.ndim != 2 or z_threshold <= 0 or not np.isfinite(z_threshold):
        raise ValueError("gray must be 2-D and z_threshold must be a finite positive value")
    if not np.isfinite(image).all():
        raise ValueError("gray must contain only finite values")
    if min(image.shape) < 3:
        return np.zeros(image.shape, dtype=bool)
    local_median = cv2.medianBlur(image, 3)
    residual = image - local_median
    mad = float(np.median(np.abs(residual - np.median(residual))))
    scale = 1.4826 * mad
    threshold = max(5.0, z_threshold * scale)
    hot = residual > threshold
    neighbor_max = cv2.dilate(local_median, np.ones((3, 3), np.uint8))
    isolated = hot & ((image - neighbor_max) > max(2.0, threshold * 0.25))
    return isolated


def denoise_gaussian(gray: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Explicit Gaussian denoising helper."""
    image = np.asarray(gray)
    if image.ndim != 2 or sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("gray must be 2-D and sigma must be a finite positive value")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("gray must contain numeric values")
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


def preprocessing_manifest(operations: list[dict[str, object]]) -> dict[str, object]:
    """Create a serializable manifest describing explicit preprocessing steps."""
    normalized = []
    for operation in operations:
        if "name" not in operation:
            raise ValueError("each preprocessing operation needs a name")
        normalized.append(dict(operation))
    return {"operations": normalized}
