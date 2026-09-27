"""Safe iteration plans for OME-TIFF, z-stacks, multichannel and time series."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile


@dataclass(frozen=True)
class LargeDataPlan:
    path: str
    series_index: int
    axes: str
    shape: tuple[int, ...]
    dtype: str
    leading_axes: str
    leading_shape: tuple[int, ...]
    plane_shape: tuple[int, int]
    strategy: str
    estimated_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


def plan_ome_iteration(path: str, *, series_index: int = 0) -> LargeDataPlan:
    source = Path(path)
    with tifffile.TiffFile(source) as tif:
        if series_index < 0 or series_index >= len(tif.series):
            raise IndexError("series_index outside available series")
        series = tif.series[series_index]
        axes, shape, dtype = str(series.axes), tuple(map(int, series.shape)), np.dtype(series.dtype)
        if not axes.endswith("YX") or len(shape) < 2:
            raise ValueError(f"safe plane streaming requires series axes ending in YX; got {axes!r}")
        leading_axes, leading_shape = axes[:-2], shape[:-2]
        nplanes = int(np.prod(leading_shape)) if leading_shape else 1
        if len(series.pages) == nplanes:
            strategy = "pages"
        else:
            try:
                mapped = tifffile.memmap(source, series=series_index)
                del mapped
                strategy = "memmap"
            except (ValueError, OSError):
                strategy = "unsupported_without_full_load"
        return LargeDataPlan(
            path=str(source.resolve()),
            series_index=series_index,
            axes=axes,
            shape=shape,
            dtype=str(dtype),
            leading_axes=leading_axes,
            leading_shape=leading_shape,
            plane_shape=(shape[-2], shape[-1]),
            strategy=strategy,
            estimated_bytes=int(np.prod(shape)) * dtype.itemsize,
        )


def iter_ome_planes(path: str, *, series_index: int = 0) -> Iterator[tuple[dict[str, int], np.ndarray]]:
    """Yield 2-D YX planes without silently loading an unsupported giant series."""
    plan = plan_ome_iteration(path, series_index=series_index)
    coords = list(np.ndindex(plan.leading_shape)) if plan.leading_shape else [()]
    if plan.strategy == "pages":
        with tifffile.TiffFile(path) as tif:
            pages = tif.series[series_index].pages
            for coord, page in zip(coords, pages):
                plane = np.asarray(page.asarray())
                if plane.shape != plan.plane_shape:
                    raise ValueError("TIFF page shape does not match planned YX plane")
                yield dict(zip(plan.leading_axes, map(int, coord))), plane
        return
    if plan.strategy == "memmap":
        mapped = tifffile.memmap(path, series=series_index)
        for coord in coords:
            plane = np.asarray(mapped[coord] if coord else mapped)
            if plane.shape != plan.plane_shape:
                raise ValueError("memory-mapped plane shape does not match plan")
            yield dict(zip(plan.leading_axes, map(int, coord))), plane
        return
    raise ValueError(
        "Series cannot be safely streamed plane-by-plane with current dependencies; "
        "convert to page-aligned OME-TIFF/Zarr or use an external whole-slide reader."
    )


def pyramid_levels(path: str) -> list[dict]:
    """Describe TIFF pyramid/sub-resolution levels without loading pixel data."""
    rows = []
    with tifffile.TiffFile(path) as tif:
        for s_idx, series in enumerate(tif.series):
            levels = getattr(series, "levels", [series])
            for level_idx, level in enumerate(levels):
                rows.append({
                    "series": s_idx,
                    "level": level_idx,
                    "axes": str(level.axes),
                    "shape": tuple(map(int, level.shape)),
                    "dtype": str(level.dtype),
                })
    return rows


__all__ = ["LargeDataPlan", "iter_ome_planes", "plan_ome_iteration", "pyramid_levels"]
