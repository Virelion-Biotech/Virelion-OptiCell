"""Replicate-aware experiment statistics exposed by the OptiCell package."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def summarize_by_replicate(features: pd.DataFrame, replicate_column: str, value_columns: Sequence[str], group_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Aggregate cell-level measurements to biological/technical replicates."""
    group_columns = list(group_columns or [])
    required = {replicate_column, *value_columns, *group_columns}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if features[replicate_column].isna().any():
        raise ValueError(f"{replicate_column} contains missing replicate IDs")
    keys = group_columns + [replicate_column]
    numeric = features[list(value_columns)].apply(pd.to_numeric, errors="coerce")
    frame = pd.concat([features[keys].reset_index(drop=True), numeric.reset_index(drop=True)], axis=1)
    agg = frame.groupby(keys, dropna=False)[list(value_columns)].agg(["mean", "median", "std", "count"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    return agg.reset_index()


def group_summary(replicate_df: pd.DataFrame, replicate_column: str, value_columns: Sequence[str], group_column: str) -> pd.DataFrame:
    """Summarize experimental groups using replicate-level observations."""
    required = {replicate_column, group_column, *value_columns}
    missing = sorted(required - set(replicate_df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if replicate_df[replicate_column].isna().any():
        raise ValueError(f"{replicate_column} contains missing replicate IDs")
    rows = []
    for group, frame in replicate_df.groupby(group_column, dropna=False):
        row = {group_column: group, "replicates": int(frame[replicate_column].nunique())}
        for col in value_columns:
            values = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{col}_mean"] = float(values.mean()) if values.size else np.nan
            row[f"{col}_median"] = float(np.median(values)) if values.size else np.nan
            row[f"{col}_std"] = float(values.std(ddof=1)) if values.size > 1 else np.nan
            row[f"{col}_sem"] = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def effect_size_mean_difference(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d for two independent replicate-level groups."""
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pooled_var = (((len(x) - 1) * x.var(ddof=1)) + ((len(y) - 1) * y.var(ddof=1))) / (len(x) + len(y) - 2)
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        raise ValueError("pooled standard deviation is zero or non-finite; Cohen's d is undefined")
    return float((x.mean() - y.mean()) / np.sqrt(pooled_var))


def permutation_pvalue(a: Sequence[float], b: Sequence[float], n_permutations: int = 10000, seed: int = 0) -> float:
    """Two-sided independent-group permutation p-value for a difference in means."""
    if n_permutations < 100:
        raise ValueError("n_permutations must be >= 100")
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    observed = abs(float(x.mean() - y.mean()))
    pooled = np.concatenate([x, y])
    rng = np.random.default_rng(seed)
    extreme = 0
    split = len(x)
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        stat = abs(float(shuffled[:split].mean() - shuffled[split:].mean()))
        extreme += stat >= observed
    return float((extreme + 1) / (n_permutations + 1))


def paired_permutation_pvalue(a: Sequence[float], b: Sequence[float], n_permutations: int = 10000, seed: int = 0) -> float:
    """Two-sided paired permutation p-value using random sign flips of pair differences."""
    if n_permutations < 100:
        raise ValueError("n_permutations must be >= 100")
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired inputs must have identical lengths")
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) == 0:
        return float("nan")
    differences = x - y
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(differences))
        stat = abs(float((differences * signs).mean()))
        extreme += stat >= observed
    return float((extreme + 1) / (n_permutations + 1))


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values."""
    p = np.asarray(list(pvalues), dtype=float)
    invalid = np.isfinite(p) & ((p < 0) | (p > 1))
    if invalid.any():
        raise ValueError("finite p-values must be in [0, 1]")
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    finite_indices = np.flatnonzero(finite)
    q[finite_indices[order]] = adjusted
    return q


def compare_two_groups(replicate_df: pd.DataFrame, value_column: str, group_column: str, group_a, group_b, n_permutations: int = 10000, seed: int = 0) -> dict[str, float]:
    """Return independent replicate-level effect size and permutation p-value."""
    if value_column not in replicate_df or group_column not in replicate_df:
        raise ValueError("value_column and group_column must exist")
    if group_a == group_b:
        raise ValueError("group_a and group_b must be distinct")
    a = pd.to_numeric(replicate_df.loc[replicate_df[group_column] == group_a, value_column], errors="coerce")
    b = pd.to_numeric(replicate_df.loc[replicate_df[group_column] == group_b, value_column], errors="coerce")
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    return {"n_group_a": float(len(a)), "n_group_b": float(len(b)), "mean_group_a": float(a.mean()) if len(a) else np.nan, "mean_group_b": float(b.mean()) if len(b) else np.nan, "mean_difference": float(a.mean() - b.mean()) if len(a) and len(b) else np.nan, "cohens_d": effect_size_mean_difference(a, b), "permutation_p": permutation_pvalue(a, b, n_permutations=n_permutations, seed=seed)}


def compare_paired_groups(pair_df: pd.DataFrame, value_column: str, group_column: str, pair_column: str, group_a, group_b, n_permutations: int = 10000, seed: int = 0) -> dict[str, float]:
    """Compare matched pairs without treating paired observations as independent."""
    required = {value_column, group_column, pair_column}
    missing = sorted(required - set(pair_df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if pair_df[pair_column].isna().any():
        raise ValueError(f"{pair_column} contains missing pair IDs")
    if group_a == group_b:
        raise ValueError("group_a and group_b must be distinct")
    if not pair_df[group_column].isin([group_a, group_b]).any():
        raise ValueError("requested groups are not present")
    selected = pair_df[pair_df[group_column].isin([group_a, group_b])].copy()
    counts = selected.groupby([pair_column, group_column], dropna=False).size()
    bad = counts[counts != 1]
    if not bad.empty:
        raise ValueError("each pair ID must occur exactly once in each requested group")
    pair_ids_a = set(selected.loc[selected[group_column] == group_a, pair_column])
    pair_ids_b = set(selected.loc[selected[group_column] == group_b, pair_column])
    if not pair_ids_a or pair_ids_a != pair_ids_b:
        raise ValueError("each pair ID must occur exactly once in each requested group")
    pivot = selected.pivot(index=pair_column, columns=group_column, values=value_column)
    values_a = pd.to_numeric(pivot[group_a], errors="coerce")
    values_b = pd.to_numeric(pivot[group_b], errors="coerce")
    finite = np.isfinite(values_a.to_numpy(float)) & np.isfinite(values_b.to_numpy(float))
    a = values_a.to_numpy(float)[finite]
    b = values_b.to_numpy(float)[finite]
    return {"n_pairs": float(len(a)), "mean_group_a": float(a.mean()) if len(a) else np.nan, "mean_group_b": float(b.mean()) if len(b) else np.nan, "mean_difference": float((a - b).mean()) if len(a) else np.nan, "paired_permutation_p": paired_permutation_pvalue(a, b, n_permutations=n_permutations, seed=seed)}
