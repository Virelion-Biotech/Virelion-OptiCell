"""Explicit acceptance rules for segmentation outputs."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SegmentationAcceptance:
    status: str
    reason: str
    score: float


def segmentation_acceptance(
    *,
    quality_score: float,
    border_fraction: float = 0.0,
    tiny_object_fraction: float = 0.0,
    merged_object_fraction: float = 0.0,
    agreement_fraction: float | None = None,
    minimum_quality: float = 70.0,
    maximum_border_fraction: float = 0.35,
    maximum_tiny_fraction: float = 0.50,
    maximum_merged_fraction: float = 0.25,
    minimum_agreement: float = 0.60,
) -> SegmentationAcceptance:
    """Classify a segmentation as PASS/REVIEW/FAIL using transparent gates."""
    values = [quality_score, border_fraction, tiny_object_fraction, merged_object_fraction]
    if any(not math.isfinite(float(v)) for v in values):
        raise ValueError("segmentation metrics must be finite")
    checks = [
        (quality_score >= minimum_quality, "quality_score below threshold"),
        (border_fraction <= maximum_border_fraction, "too many border-touching objects"),
        (tiny_object_fraction <= maximum_tiny_fraction, "too many tiny objects"),
        (merged_object_fraction <= maximum_merged_fraction, "too many merged/large objects"),
    ]
    if agreement_fraction is not None:
        if not math.isfinite(float(agreement_fraction)):
            raise ValueError("agreement_fraction must be finite")
        checks.append((agreement_fraction >= minimum_agreement, "ensemble agreement below threshold"))
    failed = [reason for passed, reason in checks if not passed]
    penalty = sum(10.0 for _ in failed)
    score = max(0.0, min(100.0, float(quality_score) - penalty))
    if not failed:
        return SegmentationAcceptance("PASS", "all configured acceptance gates passed", score)
    if quality_score < minimum_quality or len(failed) >= 3:
        return SegmentationAcceptance("FAIL", "; ".join(failed), score)
    return SegmentationAcceptance("REVIEW", "; ".join(failed), score)


__all__ = ["SegmentationAcceptance", "segmentation_acceptance"]
