"""Stable package namespace for dimension-aware image I/O and preprocessing."""
from opticell.image_io import ImageStack, canonicalize_axes, load_tiff_stack, project_z, select_channel, select_time
from opticell.preprocessing import (
    detect_hot_pixels,
    denoise_gaussian,
    estimate_background,
    flat_field_correct,
    preprocessing_manifest,
    subtract_background,
)

__all__ = [
    "ImageStack",
    "canonicalize_axes",
    "load_tiff_stack",
    "project_z",
    "select_channel",
    "select_time",
    "detect_hot_pixels",
    "denoise_gaussian",
    "estimate_background",
    "flat_field_correct",
    "preprocessing_manifest",
    "subtract_background",
]
