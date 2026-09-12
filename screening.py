"""Plate/screening statistics with explicit control definitions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _strict_numeric(values: pd.Series | np.ndarray, name: str) -> np.ndarray:
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    malformed = series.notna() & numeric.isna()
    if malformed.any():
        examples = series.loc[malformed].astype(str).head(3).tolist()
        raise ValueError(f"{name} contains non-numeric values: {examples}")
    return numeric.to_numpy(float)


def robust_zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    """Return median/MAD robust z-scores; constant inputs yield zeros."""
    x = _strict_numeric(values, "values")
    finite = np.isfinite(x)
    if not finite.any():
        return np.full(x.shape, np.nan, dtype=float)
    median = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x[finite] - median)))
    scale = 1.4826 * mad
    if scale == 0:
        scale = float(np.std(x[finite], ddof=1)) if finite.sum() > 1 else 0.0
    if scale == 0:
        return np.where(finite, 0.0, np.nan)
    return (x - median) / scale


def normalize_to_controls(
    df: pd.DataFrame,
    value_column: str,
    control_column: str,
    control_value: object,
    *,
    method: str = "median",
    output_column: str | None = None,
) -> pd.DataFrame:
    """Normalize measurements to an explicitly identified control group."""
    if value_column not in df or control_column not in df:
        raise ValueError("value_column and control_column must exist")
    values = pd.Series(_strict_numeric(df[value_column], value_column), index=df.index)
    controls = values.loc[df[control_column] == control_value].dropna()
    if controls.empty:
        raise ValueError("no valid control observations found")
    method = method.lower()
    if method == "median":
        baseline = float(controls.median())
    elif method == "mean":
        baseline = float(controls.mean())
    else:
        raise ValueError("method must be 'median' or 'mean'")
    if not np.isfinite(baseline) or baseline == 0:
        raise ValueError("control baseline must be finite and non-zero")
    result = df.copy()
    result[output_column or f"{value_column}_normalized"] = values / baseline
    result[f"{value_column}_control_baseline"] = baseline
    return result


def percent_control(
    df: pd.DataFrame,
    value_column: str,
    control_column: str,
    control_value: object,
    *,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Express values as percent of the explicit control median."""
    column = output_column or f"{value_column}_percent_control"
    normalized = normalize_to_controls(
        df, value_column, control_column, control_value,
        method="median", output_column=column,
    )
    normalized[column] *= 100.0
    return normalized


def z_prime_factor(
    control_values: pd.Series | np.ndarray,
    positive_values: pd.Series | np.ndarray,
) -> float:
    """Compute the screening Z' factor from negative and positive controls."""
    neg = _strict_numeric(control_values, "control_values")
    pos = _strict_numeric(positive_values, "positive_values")
    neg = neg[np.isfinite(neg)]
    pos = pos[np.isfinite(pos)]
    if len(neg) < 2 or len(pos) < 2:
        raise ValueError("at least two finite observations per control group are required")
    denom = abs(float(pos.mean() - neg.mean()))
    if denom == 0:
        raise ValueError("control means are identical; Z' is undefined")
    return float(1.0 - 3.0 * (float(pos.std(ddof=1)) + float(neg.std(ddof=1))) / denom)


def plate_edge_effect(df: pd.DataFrame, value_column: str, *, well_column: str = "well") -> dict[str, float]:
    """Compare edge-well and interior-well medians for an 8-by-12 plate."""
    if value_column not in df or well_column not in df:
        raise ValueError("value_column and well_column must exist")
    frame = df[[well_column, value_column]].copy()
    frame[value_column] = _strict_numeric(frame[value_column], value_column)
    frame["row"] = frame[well_column].astype(str).str[:1].str.upper()
    frame["col"] = pd.to_numeric(frame[well_column].astype(str).str[1:], errors="coerce")
    valid_wells = frame["row"].str.match(r"^[A-Z]$") & frame["col"].notna()
    if not valid_wells.all():
        raise ValueError("well identifiers must contain a row letter and numeric column")
    if not frame["col"].between(1, 12).all() or not frame["row"].isin(list("ABCDEFGH")).all():
        raise ValueError("plate_edge_effect currently requires an 8-by-12 plate with A-H rows and 1-12 columns")
    edge = frame.loc[frame["row"].isin(list("AH")) | frame["col"].isin([1, 12]), value_column].dropna()
    interior = frame.loc[~(frame["row"].isin(list("AH")) | frame["col"].isin([1, 12])), value_column].dropna()
    edge_median = float(edge.median()) if not edge.empty else np.nan
    interior_median = float(interior.median()) if not interior.empty else np.nan
    ratio = edge_median / interior_median if np.isfinite(edge_median) and interior_median != 0 else np.nan
    return {"edge_median": edge_median, "interior_median": interior_median, "edge_to_interior_ratio": float(ratio) if np.isfinite(ratio) else np.nan}


__all__ = ["robust_zscore", "normalize_to_controls", "percent_control", "z_prime_factor", "plate_edge_effect"]
