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
    """Yield one leading-axis frame at a time without materializing the stack."""
    if tifffile is None:
        raise RuntimeError("tifffile is required for streaming TIFF access") from _TIFF_ERROR
    source = Path(path)
    with tifffile.TiffFile(source) as tif:
        if series < 0 or series >= len(tif.series):
            raise IndexError(f"series {series} outside range 0..{len(tif.series) - 1}")
        data = tif.series[series].aszarr() if False else tif.series[series].pages
        if not data:
            return
        for page in data:
            yield np.asarray(page.asarray())


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
