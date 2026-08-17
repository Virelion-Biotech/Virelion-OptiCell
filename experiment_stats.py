"""Higher-level replicate-aware experiment statistics."""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def _replicate_values(frame: pd.DataFrame, value_column: str, group_column: str, replicate_column: str, group) -> np.ndarray:
    subset = frame.loc[frame[group_column] == group, [replicate_column, value_column]].copy()
    subset[value_column] = pd.to_numeric(subset[value_column], errors="coerce")
    subset = subset.dropna(subset=[value_column])
    if subset.empty:
        return np.asarray([], dtype=float)
    return subset.groupby(replicate_column, dropna=False)[value_column].mean().to_numpy(dtype=float)


def bootstrap_ci(values: Sequence[float], *, n_bootstrap: int = 5000, confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Bootstrap percentile confidence interval for a replicate-level mean."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    if not 0 < confidence < 1 or n_bootstrap < 100:
        raise ValueError("invalid confidence or n_bootstrap")
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(n_bootstrap, x.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def replicate_effect_summary(
    frame: pd.DataFrame,
    value_column: str,
    group_column: str,
    group_a,
    group_b,
    *,
    replicate_column: str = "replicate",
    confidence: float = 0.95,
    n_bootstrap: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Summarize two groups using one observation per biological/technical replicate."""
    from opticell.statistics import effect_size_mean_difference, permutation_pvalue

    a = _replicate_values(frame, value_column, group_column, replicate_column, group_a)
    b = _replicate_values(frame, value_column, group_column, replicate_column, group_b)
    low_a, high_a = bootstrap_ci(a, confidence=confidence, n_bootstrap=n_bootstrap, seed=seed)
    low_b, high_b = bootstrap_ci(b, confidence=confidence, n_bootstrap=n_bootstrap, seed=seed + 1)
    difference = float(a.mean() - b.mean()) if a.size and b.size else float("nan")
    return {
        "n_a": float(a.size),
        "n_b": float(b.size),
        "mean_a": float(a.mean()) if a.size else float("nan"),
        "mean_b": float(b.mean()) if b.size else float("nan"),
        "difference_a_minus_b": difference,
        "cohens_d": effect_size_mean_difference(a, b),
        "permutation_p": permutation_pvalue(a, b, n_permutations=10000, seed=seed),
        "a_ci_low": low_a,
        "a_ci_high": high_a,
        "b_ci_low": low_b,
        "b_ci_high": high_b,
    }


def summarize_experiment(frame: pd.DataFrame, *, replicate_column: str, group_column: str, value_columns: Sequence[str]) -> pd.DataFrame:
    """Aggregate cell/image rows to replicate level, then produce group-level means and CIs."""
    from opticell.statistics import summarize_by_replicate
    replicate = summarize_by_replicate(frame, replicate_column, value_columns, [group_column])
    rows = []
    for group, group_frame in replicate.groupby(group_column, dropna=False):
        row: dict[str, object] = {group_column: group, "replicates": int(group_frame[replicate_column].nunique())}
        for value in value_columns:
            mean_col = f"{value}_mean"
            values = pd.to_numeric(group_frame[mean_col], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            lo, hi = bootstrap_ci(values)
            row[f"{value}_mean"] = float(values.mean()) if values.size else float("nan")
            row[f"{value}_ci_low"] = lo
            row[f"{value}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["bootstrap_ci", "replicate_effect_summary", "summarize_experiment"]
