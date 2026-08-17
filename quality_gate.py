"""Transparent experiment-level quality gate composition.

The composite decision preserves its component scores and reasons. It is a QC
triage layer, not a biological validity claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExperimentQualityGate:
    status: str
    score: float
    artifact_score: float
    segmentation_score: float
    reasons: tuple[str, ...]


def experiment_quality_gate(
    *,
    artifact_score: float,
    segmentation_score: float,
    artifact_status: str | None = None,
    segmentation_status: str | None = None,
    review_threshold: float = 70.0,
    pass_threshold: float = 85.0,
) -> ExperimentQualityGate:
    """Combine acquisition and segmentation QC into PASS/REVIEW/FAIL."""
    if review_threshold < 0 or pass_threshold > 100 or review_threshold >= pass_threshold:
        raise ValueError("thresholds must satisfy 0 <= review_threshold < pass_threshold <= 100")
    scores = [float(artifact_score), float(segmentation_score)]
    if any(not math.isfinite(v) or not 0 <= v <= 100 for v in scores):
        raise ValueError("component scores must be finite values in [0, 100]")
    score = float(min(scores))
    reasons: list[str] = []
    if artifact_status == "FAIL":
        reasons.append("acquisition artifact QC failed")
    elif artifact_status == "REVIEW":
        reasons.append("acquisition artifact QC requires review")
    if segmentation_status == "FAIL":
        reasons.append("segmentation QC failed")
    elif segmentation_status == "REVIEW":
        reasons.append("segmentation QC requires review")
    if any(status == "FAIL" for status in (artifact_status, segmentation_status)) or score < review_threshold:
        status = "FAIL"
    elif any(status == "REVIEW" for status in (artifact_status, segmentation_status)) or score < pass_threshold:
        status = "REVIEW"
    else:
        status = "PASS"
    if not reasons:
        reasons.append("all configured component QC gates passed")
    return ExperimentQualityGate(status, score, scores[0], scores[1], tuple(reasons))


__all__ = ["ExperimentQualityGate", "experiment_quality_gate"]
