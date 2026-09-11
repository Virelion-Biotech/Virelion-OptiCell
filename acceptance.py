"""Explicit acceptance rules for segmentation outputs."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SegmentationAcceptance:
    status: str
    reason: str
    score: float


def _require_range(name: str, value: float, low: float, high: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return value


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
    quality_score = _require_range("quality_score", quality_score, 0.0, 100.0)
    border_fraction = _require_range("border_fraction", border_fraction, 0.0, 1.0)
    tiny_object_fraction = _require_range("tiny_object_fraction", tiny_object_fraction, 0.0, 1.0)
    merged_object_fraction = _require_range("merged_object_fraction", merged_object_fraction, 0.0, 1.0)
    minimum_quality = _require_range("minimum_quality", minimum_quality, 0.0, 100.0)
    maximum_border_fraction = _require_range("maximum_border_fraction", maximum_border_fraction, 0.0, 1.0)
    maximum_tiny_fraction = _require_range("maximum_tiny_fraction", maximum_tiny_fraction, 0.0, 1.0)
    maximum_merged_fraction = _require_range("maximum_merged_fraction", maximum_merged_fraction, 0.0, 1.0)
    if agreement_fraction is not None:
        agreement_fraction = _require_range("agreement_fraction", agreement_fraction, 0.0, 1.0)
        minimum_agreement = _require_range("minimum_agreement", minimum_agreement, 0.0, 1.0)
    elif not math.isfinite(float(minimum_agreement)) or not 0.0 <= float(minimum_agreement) <= 1.0:
        raise ValueError("minimum_agreement must be finite and between 0.0 and 1.0")

    checks = [
        (quality_score >= minimum_quality, "quality_score below threshold"),
        (border_fraction <= maximum_border_fraction, "too many border-touching objects"),
        (tiny_object_fraction <= maximum_tiny_fraction, "too many tiny objects"),
        (merged_object_fraction <= maximum_merged_fraction, "too many merged/large objects"),
    ]
    if agreement_fraction is not None:
        checks.append((agreement_fraction >= minimum_agreement, "ensemble agreement below threshold"))
    failed = [reason for passed, reason in checks if not passed]
    penalty = sum(10.0 for _ in failed)
    score = max(0.0, min(100.0, quality_score - penalty))
    if not failed:
        return SegmentationAcceptance("PASS", "all configured acceptance gates passed", score)
    if quality_score < minimum_quality or len(failed) >= 3:
        return SegmentationAcceptance("FAIL", "; ".join(failed), score)
    return SegmentationAcceptance("REVIEW", "; ".join(failed), score)


__all__ = ["SegmentationAcceptance", "segmentation_acceptance"]
