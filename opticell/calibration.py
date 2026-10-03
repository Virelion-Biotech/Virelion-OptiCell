"""Empirical calibration utilities for QC scores and acceptance thresholds.

These functions require labeled validation outcomes. They do not make an
uncalibrated OptiCell score into a probability by themselves.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _arrays(scores: Iterable[float], outcomes: Iterable[int | bool]) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(outcomes), dtype=int)
    if s.ndim != 1 or y.ndim != 1 or len(s) != len(y) or len(s) == 0:
        raise ValueError("scores and outcomes must be equally sized non-empty 1-D sequences")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("outcomes must contain only 0/1 or boolean values")
    return s, y


def calibration_curve(scores: Iterable[float], outcomes: Iterable[int | bool], *, bins: int = 10) -> pd.DataFrame:
    """Bin a 0-100 QC score and report observed success frequency."""
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    s, y = _arrays(scores, outcomes)
    if (s < 0).any() or (s > 100).any():
        raise ValueError("QC scores must be in [0,100]")
    edges = np.linspace(0.0, 100.0, bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin": b,
            "score_low": float(edges[b]),
            "score_high": float(edges[b + 1]),
            "n": int(mask.sum()),
            "mean_score": float(s[mask].mean()),
            "observed_success": float(y[mask].mean()),
        })
    return pd.DataFrame(rows)


def threshold_performance(scores: Iterable[float], outcomes: Iterable[int | bool]) -> pd.DataFrame:
    """Evaluate every distinct score as a pass threshold."""
    s, y = _arrays(scores, outcomes)
    thresholds = np.unique(np.r_[0.0, s, 100.0])
    rows = []
    for threshold in thresholds:
        pred = s >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        tn = int(np.sum((~pred) & (y == 0)))
        fn = int(np.sum((~pred) & (y == 1)))
        sensitivity = tp / (tp + fn) if tp + fn else math.nan
        specificity = tn / (tn + fp) if tn + fp else math.nan
        precision = tp / (tp + fp) if tp + fp else math.nan
        rows.append({
            "threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": sensitivity, "specificity": specificity, "precision": precision,
        })
    return pd.DataFrame(rows)


def choose_threshold(
    scores: Iterable[float],
    outcomes: Iterable[int | bool],
    *,
    minimum_sensitivity: float = 0.90,
    minimum_specificity: float = 0.80,
) -> dict[str, float | int]:
    """Choose the highest-specificity threshold meeting explicit constraints."""
    for value, name in ((minimum_sensitivity, "minimum_sensitivity"), (minimum_specificity, "minimum_specificity")):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    table = threshold_performance(scores, outcomes)
    eligible = table.loc[
        (table["sensitivity"] >= minimum_sensitivity) &
        (table["specificity"] >= minimum_specificity)
    ]
    if eligible.empty:
        raise ValueError("no threshold meets the requested sensitivity/specificity constraints")
    best = eligible.sort_values(["specificity", "sensitivity", "threshold"], ascending=[False, False, False]).iloc[0]
    return {key: (int(best[key]) if key in {"tp","fp","tn","fn"} else float(best[key])) for key in best.index}


__all__ = ["calibration_curve", "choose_threshold", "threshold_performance"]
