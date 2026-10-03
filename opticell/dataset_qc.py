"""Dataset-level QC, outlier detection, replicate summaries, and plate views."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from experiment import plate_heatmap
from experiment_qc import robust_zscore


def annotate_outliers(df: pd.DataFrame, metrics: Sequence[str], *, z_threshold: float = 3.0) -> pd.DataFrame:
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive")
    missing = sorted(set(metrics) - set(df.columns))
    if missing:
        raise ValueError(f"missing metrics: {missing}")
    result = df.copy()
    flags = pd.Series([""] * len(result), index=result.index, dtype=object)
    for metric in metrics:
        numeric = pd.to_numeric(result[metric], errors="coerce")
        malformed = result[metric].notna() & numeric.isna()
        if malformed.any():
            raise ValueError(f"{metric} contains non-numeric values")
        finite = numeric[np.isfinite(numeric)]
        z = np.full(len(result), np.nan, dtype=float)
        if len(finite):
            z[:] = robust_zscore(numeric.to_numpy(float), finite.to_numpy(float))
        result[f"{metric}_robust_z"] = z
        mask = np.isfinite(z) & (np.abs(z) > z_threshold)
        for idx in result.index[mask]:
            token = f"{metric}:OUTLIER"
            flags.loc[idx] = ";".join(filter(None, [flags.loc[idx], token]))
    result["dataset_qc_flags"] = flags
    result["dataset_qc_outlier"] = flags.ne("")
    return result


def dataset_qc_summary(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    annotated = annotate_outliers(df, metrics)
    rows = []
    for metric in metrics:
        values = pd.to_numeric(annotated[metric], errors="coerce")
        finite = values[np.isfinite(values)]
        rows.append({
            "metric": metric,
            "n": int(finite.size),
            "mean": float(finite.mean()) if finite.size else np.nan,
            "median": float(finite.median()) if finite.size else np.nan,
            "std": float(finite.std(ddof=1)) if finite.size > 1 else np.nan,
            "min": float(finite.min()) if finite.size else np.nan,
            "max": float(finite.max()) if finite.size else np.nan,
            "outlier_fraction": float(annotated[f"{metric}_robust_z"].abs().gt(3).mean()) if len(annotated) else np.nan,
        })
    return pd.DataFrame(rows)


def replicate_qc(
    df: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("condition", "replicate"),
    metrics: Sequence[str],
) -> pd.DataFrame:
    missing = sorted((set(group_columns) | set(metrics)) - set(df.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    grouped = df.groupby(list(group_columns), dropna=False)
    rows = []
    for group, part in grouped:
        group_values = group if isinstance(group, tuple) else (group,)
        row = dict(zip(group_columns, group_values))
        row["n_images"] = int(len(part))
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_cv"] = float(values.std(ddof=1) / values.mean()) if len(values) > 1 and values.mean() != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plate_view(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return plate_heatmap(df, metric, value="mean")


__all__ = ["annotate_outliers", "dataset_qc_summary", "plate_view", "replicate_qc"]
