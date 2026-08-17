"""Robustness summaries for parameter-sensitivity experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_sensitivity(table: pd.DataFrame, *, value_column: str = "object_count") -> dict[str, float | int | str]:
    """Summarize stability across parameter settings without implying correctness."""
    if value_column not in table.columns:
        raise ValueError(f"missing value column: {value_column}")
    values = pd.to_numeric(table[value_column], errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        raise ValueError("no finite sensitivity observations")
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    cv = sd / abs(mean) if mean else np.nan
    return {
        "n_settings": int(values.size),
        "mean_value": mean,
        "std_value": sd,
        "coefficient_of_variation": float(cv) if np.isfinite(cv) else np.nan,
        "range_value": float(values.max() - values.min()),
        "stability": "HIGH" if values.size > 1 and cv <= 0.05 else ("MODERATE" if values.size == 1 or cv <= 0.15 else "LOW"),
    }


def stable_parameter_subset(table: pd.DataFrame, *, value_column: str = "object_count", max_cv: float = 0.10) -> pd.DataFrame:
    """Return settings whose cumulative value is within a relative deviation gate."""
    if max_cv < 0:
        raise ValueError("max_cv must be >= 0")
    if value_column not in table.columns:
        raise ValueError(f"missing value column: {value_column}")
    values = pd.to_numeric(table[value_column], errors="coerce")
    center = float(values.median())
    if not np.isfinite(center):
        return table.iloc[0:0].copy()
    relative_deviation = (values - center).abs() / max(abs(center), 1e-12)
    result = table.copy()
    result["relative_deviation"] = relative_deviation
    return result.loc[result["relative_deviation"] <= max_cv].reset_index(drop=True)


__all__ = ["stable_parameter_subset", "summarize_sensitivity"]
