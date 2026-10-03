"""Explicit decision rules for screening assay QC."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AssayQCDecision:
    status: str
    reason: str
    z_prime: float


def classify_z_prime(z_prime: float, *, excellent: float = 0.5, marginal: float = 0.0) -> str:
    """Classify a Z' factor without implying biological validity."""
    value = float(z_prime)
    excellent_value = float(excellent)
    marginal_value = float(marginal)
    if not math.isfinite(value):
        raise ValueError("z_prime must be finite")
    if not math.isfinite(excellent_value) or not math.isfinite(marginal_value):
        raise ValueError("Z' thresholds must be finite")
    if excellent_value <= marginal_value:
        raise ValueError("excellent threshold must be greater than marginal threshold")
    if value >= excellent_value:
        return "PASS"
    if value >= marginal_value:
        return "MARGINAL"
    return "FAIL"


def assay_qc_decision(z_prime: float, *, excellent: float = 0.5, marginal: float = 0.0) -> dict[str, object]:
    """Return an auditable screening QC decision and thresholds."""
    status = classify_z_prime(z_prime, excellent=excellent, marginal=marginal)
    excellent_value = float(excellent)
    marginal_value = float(marginal)
    reasons = {
        "PASS": f"Z' >= {excellent_value:g}",
        "MARGINAL": f"{marginal_value:g} <= Z' < {excellent_value:g}",
        "FAIL": f"Z' < {marginal_value:g}",
    }
    return {
        "status": status,
        "reason": reasons[status],
        "z_prime": float(z_prime),
        "excellent_threshold": excellent_value,
        "marginal_threshold": marginal_value,
    }


__all__ = ["AssayQCDecision", "assay_qc_decision", "classify_z_prime"]
