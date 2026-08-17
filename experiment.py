"""Experiment/plate metadata extraction and group-level statistics."""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

# Require a non-letter boundary so the 'e2' in 'Plate2' cannot be mistaken for a well.
_WELL = re.compile(r"(?<![A-Za-z])([A-Ha-h])([0-9]{1,2})(?![0-9])")
_TIME = re.compile(r"(?:^|[_-])(?:t|time|tp)([0-9]+)(?:[_-]|$)", re.I)
_PLATE = re.compile(r"(?:^|[_-])(?:plate|p)([0-9]+)(?:[_-]|$)", re.I)


def parse_metadata(filename: str) -> dict[str, object]:
    """Extract common plate/well/time metadata from a filename without guessing condition labels."""
    name = str(filename)
    well = _WELL.search(name)
    time = _TIME.search(name)
    plate = _PLATE.search(name)
    row = well.group(1).upper() if well else None
    col = int(well.group(2)) if well else None
    return {
        "plate": int(plate.group(1)) if plate else None,
        "well": f"{row}{col:02d}" if row and col else None,
        "well_row": row,
        "well_col": col,
        "timepoint": int(time.group(1)) if time else None,
    }


def annotate_results(
    df: pd.DataFrame,
    filename_column: str = "filename",
    metadata_columns: Optional[dict[str, object]] = None,
) -> pd.DataFrame:
    """Add parsed plate/well/time columns, plus explicit user metadata when supplied."""
    if filename_column not in df.columns:
        raise ValueError(f"missing {filename_column!r}")
    result = df.copy()
    parsed = result[filename_column].map(parse_metadata).apply(pd.Series)
    for col in parsed.columns:
        result[col] = parsed[col]
    for key, value in (metadata_columns or {}).items():
        result[key] = value
    return result


def summarize_groups(df: pd.DataFrame, group_by: list[str], metrics: list[str]) -> pd.DataFrame:
    """Summarize image/cell-level measurements by explicit experimental groups."""
    missing = (set(group_by) | set(metrics)) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    grouped = df.groupby(group_by, dropna=False)
    aggregations = {m: ["count", "mean", "median", "std"] for m in metrics}
    out = grouped.agg(aggregations)
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    return out.reset_index()


def compare_groups(df: pd.DataFrame, group_column: str, metric: str) -> pd.DataFrame:
    """Descriptive two-group comparison; no hidden inferential assumptions."""
    if group_column not in df or metric not in df:
        raise ValueError("group_column and metric must exist")
    groups = [g for g in df[group_column].dropna().unique()]
    rows = []
    for g in groups:
        values = pd.to_numeric(df.loc[df[group_column] == g, metric], errors="coerce").dropna().to_numpy(float)
        rows.append(
            {
                "group": g,
                "n": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plate_heatmap(df: pd.DataFrame, metric: str, value: str = "mean") -> pd.DataFrame:
    """Return an 8x12-like well matrix for plate QC/phenotyping."""
    required = {"well", metric}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(required - set(df.columns))}")
    tmp = df.copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    agg = tmp.groupby("well", dropna=True)[metric]
    vals = getattr(agg, value)() if value in {"mean", "median", "max", "min"} else agg.mean()
    matrix = pd.DataFrame(index=list("ABCDEFGH"), columns=range(1, 13), dtype=float)
    for well, val in vals.items():
        if isinstance(well, str) and len(well) >= 2 and well[0] in matrix.index:
            col = int(well[1:])
            if 1 <= col <= 12:
                matrix.loc[well[0], col] = float(val)
    return matrix
