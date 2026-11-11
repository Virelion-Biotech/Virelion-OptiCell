"""Transparent experiment-level quality gate composition."""
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
    """Compose transparent QC gates using the conservative minimum score."""
    thresholds = (float(review_threshold), float(pass_threshold))
    if (
        any(not math.isfinite(value) for value in thresholds)
        or not 0 <= thresholds[0] < thresholds[1] <= 100
    ):
        raise ValueError("thresholds must satisfy 0 <= review_threshold < pass_threshold <= 100")

    allowed_statuses = {None, "PASS", "REVIEW", "FAIL"}
    if artifact_status not in allowed_statuses or segmentation_status not in allowed_statuses:
        raise ValueError("component status must be None, 'PASS', 'REVIEW', or 'FAIL'")

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

    if score < thresholds[0]:
        reasons.append(f"conservative QC score below fail threshold ({thresholds[0]:g})")
        status = "FAIL"
    elif score < thresholds[1]:
        reasons.append(f"conservative QC score below pass threshold ({thresholds[1]:g})")
        status = "REVIEW"
    elif any(status == "FAIL" for status in (artifact_status, segmentation_status)):
        status = "FAIL"
    elif any(status == "REVIEW" for status in (artifact_status, segmentation_status)):
        status = "REVIEW"
    else:
        status = "PASS"
    if not reasons:
        reasons.append("all configured component QC gates passed")
    return ExperimentQualityGate(status, score, scores[0], scores[1], tuple(reasons))


__all__ = ["ExperimentQualityGate", "experiment_quality_gate"]
