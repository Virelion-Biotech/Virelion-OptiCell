"""Memory-conscious TIFF access for large microscopy datasets."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError as exc:  # pragma: no cover
    tifffile = None
    _TIFF_ERROR = exc
else:
    _TIFF_ERROR = None


def memmap_tiff(path: str, *, series: int = 0) -> np.memmap:
    """Memory-map an uncompressed TIFF series when tifffile permits it."""
    if tifffile is None:
        raise RuntimeError("tifffile is required for memory-mapped TIFF access") from _TIFF_ERROR
    source = Path(path)
    with tifffile.TiffFile(source) as tif:
        if series < 0 or series >= len(tif.series):
            raise IndexError(f"series {series} outside range 0..{len(tif.series) - 1}")
        target = tif.series[series]
        shape = target.shape
        dtype = target.dtype
    try:
        mapped = tifffile.memmap(source, series=series)
    except ValueError as exc:
        raise ValueError("TIFF series cannot be memory-mapped; use iter_tiff_frames instead") from exc
    if mapped.shape != shape or mapped.dtype != dtype:
        raise RuntimeError("memory-mapped TIFF metadata changed unexpectedly")
    return mapped


def iter_tiff_frames(path: str, *, series: int = 0, axis: int = 0) -> Iterator[np.ndarray]:
    """Yield logical frames along the leading series axis without loading the stack.

    This is supported only when tifffile exposes one TIFF page per logical
    leading-axis frame.  Series such as a ``TZYX`` stack are supported when
    each page is already a complete ``ZYX`` frame.  Ambiguous page layouts are
    rejected instead of silently yielding planes as if they were frames.
    """
    if tifffile is None:
        raise RuntimeError("tifffile is required for streaming TIFF access") from _TIFF_ERROR
    source = Path(path)
    with tifffile.TiffFile(source) as tif:
        if series < 0 or series >= len(tif.series):
            raise IndexError(f"series {series} outside range 0..{len(tif.series) - 1}")
        series_data = tif.series[series]
        if not isinstance(axis, int):
            raise TypeError("axis must be an integer")
        normalized_axis = axis + len(series_data.shape) if axis < 0 else axis
        if normalized_axis != 0:
            raise ValueError("iter_tiff_frames currently supports only the leading series axis (axis=0)")
        if not series_data.pages:
            return

        expected_shape = tuple(series_data.shape[1:])
        expected_dtype = np.dtype(series_data.dtype)
        expected_pages = int(series_data.shape[0])
        actual_pages = len(series_data.pages)
        if actual_pages != expected_pages:
            raise ValueError(
                "TIFF series page layout is ambiguous for axis-0 streaming: "
                f"series shape {series_data.shape} implies {expected_pages} pages, "
                f"but tifffile exposes {actual_pages} pages"
            )

        for index, page in enumerate(series_data.pages):
            frame = np.asarray(page.asarray())
            if tuple(frame.shape) != expected_shape or np.dtype(frame.dtype) != expected_dtype:
                raise ValueError(
                    "TIFF page does not represent one logical leading-axis frame: "
                    f"page {index} has shape/dtype {frame.shape}/{frame.dtype}, "
                    f"expected {expected_shape}/{expected_dtype}"
                )
            yield frame


def iter_array_chunks(array: np.ndarray, *, axis: int = 0, chunk_size: int = 1) -> Iterator[np.ndarray]:
    """Iterate an in-memory array in bounded chunks along one axis."""
    arr = np.asarray(array)
    if arr.ndim == 0:
        raise ValueError("array must have at least one dimension")
    if not 0 <= axis < arr.ndim:
        raise ValueError("axis is outside array dimensions")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, arr.shape[axis], chunk_size):
        stop = min(start + chunk_size, arr.shape[axis])
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(start, stop)
        yield arr[tuple(slices)]


__all__ = ["memmap_tiff", "iter_tiff_frames", "iter_array_chunks"]
