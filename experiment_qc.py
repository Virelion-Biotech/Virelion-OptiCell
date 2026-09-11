"""Control normalization and plate-level QC utilities."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def robust_zscore(values: pd.Series | np.ndarray, reference: pd.Series | np.ndarray) -> np.ndarray:
    """Compute robust z-scores using reference median and MAD."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").to_numpy(float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.full(x.shape, np.nan, dtype=float)
    median = float(np.median(ref))
    mad = float(np.median(np.abs(ref - median)))
    scale = 1.4826 * mad
    if scale == 0:
        std = float(np.std(ref, ddof=1)) if ref.size > 1 else 0.0
        scale = std if std > 0 else 0.0
    if scale == 0:
        return np.where(np.isfinite(x), 0.0, np.nan)
    return (x - median) / scale


def normalize_to_controls(
    df: pd.DataFrame,
    metric: str,
    *,
    group_column: str,
    control_value: object,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Add robust control-normalized values using only the specified control group."""
    required = {metric, group_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    result = df.copy()
    controls = result.loc[result[group_column] == control_value, metric]
    column = output_column or f"{metric}_robust_z"
    result[column] = robust_zscore(result[metric], controls)
    return result


def plate_edge_effect(
    df: pd.DataFrame,
    metric: str,
    *,
    row_column: str = "well_row",
    col_column: str = "well_col",
    edge_rows: Sequence[str] = tuple("AH"),
    edge_cols: Sequence[int] = (1, 12),
) -> dict[str, float]:
    """Compare edge wells with interior wells using median difference and robust effect."""
    required = {metric, row_column, col_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    values = pd.to_numeric(df[metric], errors="coerce")
    edge_mask = df[row_column].isin(edge_rows) | df[col_column].isin(edge_cols)
    edge = values[edge_mask].dropna().to_numpy(float)
    interior = values[~edge_mask].dropna().to_numpy(float)
    if edge.size == 0 or interior.size == 0:
        return {"edge_n": float(edge.size), "interior_n": float(interior.size), "edge_median": np.nan, "interior_median": np.nan, "median_difference": np.nan, "robust_effect": np.nan}
    pooled_median = float(np.median(interior))
    mad = float(np.median(np.abs(interior - pooled_median)))
    scale = 1.4826 * mad
    if scale == 0 and interior.size > 1:
        std = float(np.std(interior, ddof=1))
        scale = std if std > 0 else 0.0
    edge_median = float(np.median(edge))
    effect = (edge_median - pooled_median) / scale if scale > 0 else np.nan
    return {
        "edge_n": float(edge.size), "interior_n": float(interior.size),
        "edge_median": edge_median, "interior_median": pooled_median,
        "median_difference": edge_median - pooled_median,
        "robust_effect": float(effect) if np.isfinite(effect) else np.nan,
    }


def plate_qc_summary(df: pd.DataFrame, metrics: Sequence[str], *, control_group: str | None = None, control_value: object | None = None, group_column: str = "condition") -> pd.DataFrame:
    """Produce a compact plate-level QC table with control-reference dispersion and outliers."""
    missing = sorted(set(metrics) - set(df.columns))
    if missing:
        raise ValueError(f"missing metric columns: {missing}")
    if control_group is not None and control_group not in df.columns:
        raise ValueError(f"missing control group column: {control_group}")
    rows = []
    for metric in metrics:
        values = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy(float)
        row = {"metric": metric, "n": int(values.size), "mean": float(values.mean()) if values.size else np.nan, "median": float(np.median(values)) if values.size else np.nan, "std": float(values.std(ddof=1)) if values.size > 1 else np.nan}
        if control_group is not None and control_value is not None:
            controls = pd.to_numeric(df.loc[df[control_group] == control_value, metric], errors="coerce").dropna().to_numpy(float)
            if controls.size == 0:
                raise ValueError(f"no valid control observations found for metric {metric!r}")
            control_median = float(np.median(controls))
            control_mad = float(np.median(np.abs(controls - control_median)))
            control_scale = 1.4826 * control_mad
            if control_scale == 0 and controls.size > 1:
                control_std = float(np.std(controls, ddof=1))
                control_scale = control_std if control_std > 0 else 0.0
            z = robust_zscore(values, controls)
            finite = z[np.isfinite(z)]
            row["control_robust_sd"] = float(control_scale) if control_scale > 0 else np.nan
            row["control_outlier_fraction_abs_z_gt_3"] = float(np.mean(np.abs(finite) > 3)) if finite.size else np.nan
        else:
            row["control_robust_sd"] = np.nan
            row["control_outlier_fraction_abs_z_gt_3"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["plate_edge_effect", "plate_qc_summary", "normalize_to_controls", "robust_zscore"]
