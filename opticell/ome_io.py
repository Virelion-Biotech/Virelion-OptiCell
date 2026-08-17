"""OME-TIFF metadata helpers without changing image pixels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import tifffile


@dataclass(frozen=True)
class OMEImageInfo:
    path: str
    axes: str
    shape: tuple[int, ...]
    size_x: int | None
    size_y: int | None
    size_z: int | None
    size_c: int | None
    size_t: int | None
    physical_size_x: float | None
    physical_size_y: float | None
    physical_size_z: float | None
    physical_unit_x: str | None
    physical_unit_y: str | None
    physical_unit_z: str | None
    channel_names: tuple[str, ...]


def _float_attr(node: ET.Element | None, name: str) -> float | None:
    value = node.attrib.get(name) if node is not None else None
    return float(value) if value not in (None, "") else None


def read_ome_info(path: str) -> OMEImageInfo:
    """Read dimensions, physical scales, units, and channel names from OME metadata."""
    source = Path(path)
    if source.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("read_ome_info expects a .tif or .tiff file")
    with tifffile.TiffFile(source) as tif:
        if not tif.series:
            raise ValueError(f"No TIFF series found in {source}")
        series = tif.series[0]
        axes = str(series.axes)
        shape = tuple(int(x) for x in series.shape)
        ome_xml = tif.ome_metadata
    sizes = {axis: shape[axes.index(axis)] if axis in axes else None for axis in "XYZCT"}
    physical = {axis: None for axis in "XYZ"}
    units = {axis: None for axis in "XYZ"}
    channel_names: tuple[str, ...] = ()
    if ome_xml:
        root = ET.fromstring(ome_xml)
        pixels = root.find(".//{*}Pixels")
        if pixels is not None:
            for axis in physical:
                physical[axis] = _float_attr(pixels, f"PhysicalSize{axis}")
                units[axis] = pixels.attrib.get(f"PhysicalSize{axis}Unit")
            channel_names = tuple(ch.attrib.get("Name", "") for ch in pixels.findall("{*}Channel"))
    return OMEImageInfo(
        path=str(source.resolve()), axes=axes, shape=shape,
        size_x=sizes["X"], size_y=sizes["Y"], size_z=sizes["Z"], size_c=sizes["C"], size_t=sizes["T"],
        physical_size_x=physical["X"], physical_size_y=physical["Y"], physical_size_z=physical["Z"],
        physical_unit_x=units["X"], physical_unit_y=units["Y"], physical_unit_z=units["Z"],
        channel_names=channel_names,
    )


def load_ome_series(path: str, *, series_index: int = 0) -> tuple[np.ndarray, str]:
    """Load a TIFF series and return its pixel array and declared axes."""
    if series_index < 0:
        raise ValueError("series_index must be non-negative")
    with tifffile.TiffFile(path) as tif:
        if series_index >= len(tif.series):
            raise IndexError(f"series_index {series_index} outside available series")
        series = tif.series[series_index]
        return np.asarray(series.asarray()), str(series.axes)


__all__ = ["OMEImageInfo", "load_ome_series", "read_ome_info"]
