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
    robust_scale. Missing values remain missing. This is descriptive batch correction,
    not a substitute for a prespecified experimental model.
    """
    required = {value_column, row_column, column_column}
    if not required.issubset(df.columns):
        raise ValueError(f"missing required columns: {sorted(required - set(df.columns))}")
    out = df.copy()
    x = pd.to_numeric(out[value_column], errors="coerce")
    finite = x.notna()
    if not finite.any():
        out[output_column or f"{value_column}_bscore"] = np.nan
        return out
    grand = float(x[finite].median())
    row_med = x.groupby(out[row_column], dropna=False).transform("median")
    col_med = x.groupby(out[column_column], dropna=False).transform("median")
    corrected = x - row_med - col_med + grand
    mad = float(np.nanmedian(np.abs(corrected.to_numpy(float)[np.isfinite(corrected)])))
    scale = 1.4826 * mad
    if scale == 0:
        scale = float(np.nanstd(corrected.to_numpy(float)))
    if scale == 0 or not np.isfinite(scale):
        scale = 1.0
    out[output_column or f"{value_column}_bscore"] = corrected / scale
    return out


def ssmd(
    control_values: pd.Series | np.ndarray,
    treatment_values: pd.Series | np.ndarray,
) -> float:
    """Return strictly standardized mean difference for two independent groups."""
    a = pd.to_numeric(pd.Series(control_values), errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(pd.Series(treatment_values), errors="coerce").dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("at least two observations per group are required")
    pooled = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
    if pooled == 0:
        raise ValueError("pooled standard deviation is zero; SSMD is undefined")
    return float((b.mean() - a.mean()) / pooled)


def plate_uniformity(values: pd.Series | np.ndarray) -> dict[str, float]:
    """Summarize robust plate dispersion using median, MAD, and robust CV."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if x.size == 0:
        return {"n": 0.0, "median": np.nan, "mad": np.nan, "robust_cv": np.nan}
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    robust_sd = 1.4826 * mad
    robust_cv = robust_sd / abs(median) if median != 0 else np.nan
    return {"n": float(x.size), "median": median, "mad": mad, "robust_cv": float(robust_cv) if np.isfinite(robust_cv) else np.nan}


__all__ = ["b_score", "ssmd", "plate_uniformity"]
