"""Native microscopy dimension-aware I/O utilities.

Unlike the legacy 2-D analysis loader, this module preserves TIFF axes and
provides explicit C/Z/T selection/projection helpers. It avoids guessing when
metadata are available and raises on ambiguous requests instead of silently
collapsing scientifically meaningful dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import tifffile
except ImportError as exc:  # pragma: no cover
    tifffile = None
    _TIFF_ERROR = exc
else:
    _TIFF_ERROR = None


@dataclass(frozen=True)
class ImageStack:
    data: np.ndarray
    axes: str
    path: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def has_channel(self) -> bool:
        return "C" in self.axes

    @property
    def has_z(self) -> bool:
        return "Z" in self.axes

    @property
    def has_time(self) -> bool:
        return "T" in self.axes


def load_tiff_stack(path: str) -> ImageStack:
    """Read the first TIFF series while preserving its declared axes."""
    if tifffile is None:
        raise RuntimeError("tifffile is required for native stack I/O") from _TIFF_ERROR
    source = Path(path)
    if source.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("load_tiff_stack expects a .tif or .tiff file")
    with tifffile.TiffFile(source) as tif:
        if not tif.series:
            raise ValueError(f"No TIFF series found in {source}")
        series = tif.series[0]
        data = np.asarray(series.asarray())
        axes = str(series.axes)
    if len(axes) != data.ndim:
        raise ValueError(f"TIFF axes {axes!r} do not match data shape {data.shape}")
    return ImageStack(data=data, axes=axes, path=str(source.resolve()), dtype=str(data.dtype), shape=data.shape)


def canonicalize_axes(stack: ImageStack) -> ImageStack:
    """Reorder dimensions to T,Z,C,Y,X where those axes exist.

    Missing dimensions are left absent; unknown axes are rejected rather than
    silently reassigned.
    """
    allowed = set("TZCYX")
    unknown = set(stack.axes) - allowed
    if unknown:
        raise ValueError(f"Unsupported TIFF axes: {sorted(unknown)}")
    order = [axis for axis in "TZCYX" if axis in stack.axes]
    transpose = [stack.axes.index(axis) for axis in order]
    data = np.transpose(stack.data, transpose) if transpose != list(range(stack.data.ndim)) else stack.data
    return ImageStack(data=data, axes="".join(order), path=stack.path, dtype=str(data.dtype), shape=data.shape)


def select_channel(stack: ImageStack, channel: int = 0) -> np.ndarray:
    """Extract one channel without altering Z/T structure."""
    if "C" not in stack.axes:
        if channel != 0:
            raise IndexError("Channel axis is absent; only channel=0 is valid")
        return stack.data
    axis = stack.axes.index("C")
    if not 0 <= channel < stack.data.shape[axis]:
        raise IndexError(f"channel {channel} outside range 0..{stack.data.shape[axis]-1}")
    return np.take(stack.data, channel, axis=axis)


def project_z(stack: ImageStack, method: str = "max") -> ImageStack:
    """Project Z while preserving all other dimensions."""
    if "Z" not in stack.axes:
        return stack
    axis = stack.axes.index("Z")
    method = method.lower()
    if method == "max":
        data = np.max(stack.data, axis=axis)
    elif method == "mean":
        data = np.mean(stack.data, axis=axis).astype(stack.data.dtype, copy=False)
    elif method == "median":
        data = np.median(stack.data, axis=axis).astype(stack.data.dtype, copy=False)
    else:
        raise ValueError("method must be one of: max, mean, median")
    axes = stack.axes.replace("Z", "")
    return ImageStack(data=data, axes=axes, path=stack.path, dtype=str(data.dtype), shape=data.shape)


def select_time(stack: ImageStack, timepoint: int = 0) -> ImageStack:
    """Select one timepoint while preserving remaining dimensions."""
    if "T" not in stack.axes:
        if timepoint != 0:
            raise IndexError("Time axis is absent; only timepoint=0 is valid")
        return stack
    axis = stack.axes.index("T")
    if not 0 <= timepoint < stack.data.shape[axis]:
        raise IndexError(f"timepoint {timepoint} outside range 0..{stack.data.shape[axis]-1}")
    data = np.take(stack.data, timepoint, axis=axis)
    axes = stack.axes.replace("T", "")
    return ImageStack(data=data, axes=axes, path=stack.path, dtype=str(data.dtype), shape=data.shape)
