"""Explicit, provenance-friendly microscopy preprocessing operations."""
from __future__ import annotations

import cv2
import numpy as np


def flat_field_correct(image: np.ndarray, reference: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Correct multiplicative illumination using a reference field."""
    image = np.asarray(image, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if image.shape != reference.shape:
        raise ValueError("image and reference must have identical shapes")
    if np.any(reference <= 0):
        raise ValueError("reference must contain strictly positive values")
    scale = float(reference.mean())
    corrected = image * scale / np.maximum(reference, eps)
    if np.issubdtype(np.asarray(reference).dtype, np.integer):
        info = np.iinfo(np.asarray(reference).dtype)
        corrected = np.clip(corrected, info.min, info.max).astype(reference.dtype)
    return corrected


def estimate_background(gray: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Estimate smooth illumination/background without altering the source."""
    image = np.asarray(gray)
    if image.ndim != 2 or sigma <= 0:
        raise ValueError("gray must be 2-D and sigma must be > 0")
    work = image.astype(np.float32, copy=False)
    return cv2.GaussianBlur(work, (0, 0), sigmaX=sigma)


def subtract_background(gray: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Subtract estimated background while preserving the input numeric dtype when possible."""
    image = np.asarray(gray)
    background = estimate_background(image, sigma=sigma)
    corrected = np.clip(image.astype(np.float32) - background, 0, None)
    if image.dtype == np.uint8:
        return np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected.astype(image.dtype, copy=False)


def detect_hot_pixels(gray: np.ndarray, z_threshold: float = 6.0) -> np.ndarray:
    """Return a boolean mask of unusually bright isolated pixels."""
    image = np.asarray(gray, dtype=np.float32)
    if image.ndim != 2 or z_threshold <= 0:
        raise ValueError("gray must be 2-D and z_threshold must be > 0")
    median = float(np.median(image))
    mad = float(np.median(np.abs(image - median)))
    robust_scale = 1.4826 * mad
    if robust_scale == 0:
        return np.zeros(image.shape, dtype=bool)
    return image > median + z_threshold * robust_scale


def denoise_gaussian(gray: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Explicit Gaussian denoising helper."""
    image = np.asarray(gray)
    if image.ndim != 2 or sigma <= 0:
        raise ValueError("gray must be 2-D and sigma must be > 0")
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


def preprocessing_manifest(operations: list[dict[str, object]]) -> dict[str, object]:
    """Create a serializable manifest describing explicit preprocessing steps."""
    normalized = []
    for operation in operations:
        if "name" not in operation:
            raise ValueError("each preprocessing operation needs a name")
        normalized.append(dict(operation))
    return {"operations": normalized}
