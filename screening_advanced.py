"""Robust plate-screening statistics for batch-effect-aware assays."""
from __future__ import annotations

import numpy as np
import pandas as pd


def b_score(
    df: pd.DataFrame,
    value_column: str,
    *,
    row_column: str = "row",
    column_column: str = "column",
    output_column: str | None = None,
) -> pd.DataFrame:
    """Apply a robust two-way row/column median correction (B-score style).

    The returned score is (value - row_effect - column_effect + grand_median) /
    robust_scale. Missing values in the measurement remain missing. Plate
    coordinates must be present for every measured value; silently treating
    missing coordinates as a shared batch would make the correction ambiguous.
    This is descriptive batch correction, not a substitute for a prespecified
    experimental model.
    """
    required = {value_column, row_column, column_column}
    if not required.issubset(df.columns):
        raise ValueError(f"missing required columns: {sorted(required - set(df.columns))}")
    out = df.copy()
    x = pd.to_numeric(out[value_column], errors="coerce")
    if np.isinf(x.to_numpy(float)).any():
        raise ValueError("measurement values must be finite or missing")
    coordinate_missing = out.loc[x.notna(), [row_column, column_column]].isna().any(axis=1)
    if coordinate_missing.any():
        raise ValueError("row/column coordinates cannot be missing for measured values")
    finite = x.notna()
    if not finite.any():
        out[output_column or f"{value_column}_bscore"] = np.nan
        return out
    grand = float(x[finite].median())
    row_med = x.groupby(out[row_column], dropna=False).transform("median")
    col_med = x.groupby(out[column_column], dropna=False).transform("median")
    corrected = x - row_med - col_med + grand
    finite_corrected = corrected.to_numpy(float)
    finite_corrected = finite_corrected[np.isfinite(finite_corrected)]
    mad = float(np.median(np.abs(finite_corrected)))
    scale = 1.4826 * mad
    if scale == 0:
        scale = float(np.std(finite_corrected, ddof=1)) if finite_corrected.size > 1 else 0.0
    if scale == 0:
        # A constant plate has no measurable dispersion; returning zero scores
        # is mathematically well-defined for the centered values.
        scaled = corrected * 0.0
    elif np.isfinite(scale):
        scaled = corrected / scale
    else:
        raise ValueError("could not compute a finite B-score scale")
    out[output_column or f"{value_column}_bscore"] = scaled
    return out


def ssmd(
    control_values: pd.Series | np.ndarray,
    treatment_values: pd.Series | np.ndarray,
) -> float:
    """Return the strictly standardized mean difference for two independent groups.

    SSMD uses sqrt(var_control + var_treatment) as its denominator. This is
    distinct from Cohen's d, which uses a pooled standard deviation.
    """
    a = pd.to_numeric(pd.Series(control_values), errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(pd.Series(treatment_values), errors="coerce").dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("at least two observations per group are required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("group values must be finite")
    denominator = float(np.sqrt(a.var(ddof=1) + b.var(ddof=1)))
    if denominator == 0:
        raise ValueError("combined variance is zero; SSMD is undefined")
    return float((b.mean() - a.mean()) / denominator)


def plate_uniformity(values: pd.Series | np.ndarray) -> dict[str, float]:
    """Summarize robust plate dispersion using median, MAD, and robust CV."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("plate values must be finite or missing")
    if x.size == 0:
        return {"n": 0.0, "median": np.nan, "mad": np.nan, "robust_cv": np.nan}
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    robust_sd = 1.4826 * mad
    robust_cv = robust_sd / abs(median) if median != 0 else np.nan
    return {"n": float(x.size), "median": median, "mad": mad, "robust_cv": float(robust_cv) if np.isfinite(robust_cv) else np.nan}


__all__ = ["b_score", "ssmd", "plate_uniformity"]
