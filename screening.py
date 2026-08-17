"""Plate/screening statistics with explicit control definitions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def robust_zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    """Return median/MAD robust z-scores; constant inputs yield zeros."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.full(x.shape, np.nan, dtype=float)
    median = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x[finite] - median)))
    scale = 1.4826 * mad
    if scale == 0:
        scale = float(np.nanstd(x[finite]))
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
    values = pd.to_numeric(df[value_column], errors="coerce")
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
    if baseline == 0:
        raise ValueError("control baseline is zero; division normalization is undefined")
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
    normalized = normalize_to_controls(
        df,
        value_column,
        control_column,
        control_value,
        method="median",
        output_column=output_column or f"{value_column}_fraction_control",
    )
    normalized[output_column or f"{value_column}_fraction_control"] *= 100.0
    return normalized


def z_prime_factor(
    control_values: pd.Series | np.ndarray,
    positive_values: pd.Series | np.ndarray,
) -> float:
    """Compute the screening Z' factor from negative and positive controls."""
    neg = pd.to_numeric(pd.Series(control_values), errors="coerce").dropna().to_numpy(float)
    pos = pd.to_numeric(pd.Series(positive_values), errors="coerce").dropna().to_numpy(float)
    if len(neg) < 2 or len(pos) < 2:
        raise ValueError("at least two observations per control group are required")
    denom = abs(float(pos.mean() - neg.mean()))
    if denom == 0:
        raise ValueError("control means are identical; Z' is undefined")
    return float(1.0 - 3.0 * (float(pos.std(ddof=1)) + float(neg.std(ddof=1))) / denom)


def plate_edge_effect(df: pd.DataFrame, value_column: str, *, well_column: str = "well") -> dict[str, float]:
    """Compare edge-well and interior-well medians without assuming direction."""
    if value_column not in df or well_column not in df:
        raise ValueError("value_column and well_column must exist")
    frame = df[[well_column, value_column]].copy()
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame["row"] = frame[well_column].astype(str).str[:1].str.upper()
    frame["col"] = pd.to_numeric(frame[well_column].astype(str).str[1:], errors="coerce")
    edge = frame.loc[frame["row"].isin(list("AH")) | frame["col"].isin([1, 12]), value_column].dropna()
    interior = frame.loc[~(frame["row"].isin(list("AH")) | frame["col"].isin([1, 12])), value_column].dropna()
    edge_median = float(edge.median()) if not edge.empty else np.nan
    interior_median = float(interior.median()) if not interior.empty else np.nan
    ratio = edge_median / interior_median if np.isfinite(edge_median) and interior_median != 0 else np.nan
    return {"edge_median": edge_median, "interior_median": interior_median, "edge_to_interior_ratio": float(ratio) if np.isfinite(ratio) else np.nan}


__all__ = ["robust_zscore", "normalize_to_controls", "percent_control", "z_prime_factor", "plate_edge_effect"]
