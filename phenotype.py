"""Deterministic, auditable cell-phenotype scoring utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    feature: str
    threshold: float
    direction: str = ">="
    weight: float = 1.0
    label: str = "positive"

    def __post_init__(self) -> None:
        if not self.feature:
            raise ValueError("feature must be non-empty")
        if self.direction not in {">=", "<="}:
            raise ValueError("direction must be >= or <=")
        if not np.isfinite(float(self.threshold)):
            raise ValueError("threshold must be finite")
        if not np.isfinite(float(self.weight)) or self.weight < 0:
            raise ValueError("weight must be a finite non-negative value")
        if not self.label:
            raise ValueError("label must be non-empty")

    def evaluate(self, value: float) -> bool:
        if not np.isfinite(value):
            return False
        if self.direction == ">=":
            return value >= self.threshold
        return value <= self.threshold


def score_cells(
    features: pd.DataFrame,
    rules: list[Rule],
    positive_label: str = "positive",
    negative_label: str = "negative",
) -> pd.DataFrame:
    """Score cells without a black-box model; every contribution is explicit."""
    if not positive_label or not negative_label or positive_label == negative_label:
        raise ValueError("positive_label and negative_label must be distinct non-empty strings")
    result = features.copy()
    scores = np.zeros(len(result), dtype=float)
    reasons = [[] for _ in range(len(result))]
    for rule in rules:
        if rule.feature not in result.columns:
            raise ValueError(f"feature {rule.feature!r} not present")
        passed = pd.to_numeric(result[rule.feature], errors="coerce").map(rule.evaluate).to_numpy(dtype=bool)
        scores += passed.astype(float) * float(rule.weight)
        for i, ok in enumerate(passed):
            if ok:
                reasons[i].append(rule.label)
    max_score = float(sum(r.weight for r in rules))
    result["phenotype_score"] = scores
    result["phenotype_score_fraction"] = scores / max_score if max_score > 0 else 0.0
    result["phenotype_label"] = np.where(scores > 0, positive_label, negative_label)
    result["phenotype_reasons"] = ["; ".join(r) for r in reasons]
    return result


def marker_positivity(
    features: pd.DataFrame,
    intensity_column: str,
    threshold: float,
    positive_label: str = "positive",
    negative_label: str = "negative",
) -> pd.DataFrame:
    """Call marker-positive cells from an explicit intensity cutoff."""
    if intensity_column not in features.columns:
        raise ValueError(f"missing {intensity_column!r}")
    if not np.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    if not positive_label or not negative_label or positive_label == negative_label:
        raise ValueError("positive_label and negative_label must be distinct non-empty strings")
    result = features.copy()
    values = pd.to_numeric(result[intensity_column], errors="coerce")
    result["marker_threshold"] = float(threshold)
    result["marker_positive"] = values >= threshold
    result["marker_label"] = np.where(result["marker_positive"], positive_label, negative_label)
    return result


def group_phenotype_summary(
    features: pd.DataFrame,
    group_column: Optional[str] = None,
    positive_label: str = "positive",
) -> pd.DataFrame:
    """Summarize positive fraction and score distributions by optional group."""
    frame = features.copy()
    if "phenotype_label" not in frame or "phenotype_score" not in frame:
        raise ValueError("run score_cells before summarizing")
    if not positive_label:
        raise ValueError("positive_label must be non-empty")
    if group_column and group_column not in frame.columns:
        raise ValueError(f"missing {group_column!r}")
    if group_column:
        grouped = frame.groupby(group_column, dropna=False)
        return grouped.agg(
            cell_count=("phenotype_score", "size"),
            positive_fraction=("phenotype_label", lambda s: float((s == positive_label).mean())),
            mean_score=("phenotype_score", "mean"),
            median_score=("phenotype_score", "median"),
        ).reset_index()
    return pd.DataFrame([{
        "cell_count": len(frame),
        "positive_fraction": float((frame["phenotype_label"] == positive_label).mean()) if len(frame) else 0.0,
        "mean_score": float(frame["phenotype_score"].mean()) if len(frame) else np.nan,
        "median_score": float(frame["phenotype_score"].median()) if len(frame) else np.nan,
    }])
