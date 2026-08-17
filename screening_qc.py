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
    if not math.isfinite(value):
        raise ValueError("z_prime must be finite")
    if excellent <= marginal:
        raise ValueError("excellent threshold must be greater than marginal threshold")
    if value >= excellent:
        return "PASS"
    if value >= marginal:
        return "MARGINAL"
    return "FAIL"


def assay_qc_decision(z_prime: float, *, excellent: float = 0.5, marginal: float = 0.0) -> dict[str, object]:
    """Return an auditable screening QC decision and thresholds."""
    status = classify_z_prime(z_prime, excellent=excellent, marginal=marginal)
    reasons = {
        "PASS": f"Z' >= {excellent:g}",
        "MARGINAL": f"{marginal:g} <= Z' < {excellent:g}",
        "FAIL": f"Z' < {marginal:g}",
    }
    return {
        "status": status,
        "reason": reasons[status],
        "z_prime": float(z_prime),
        "excellent_threshold": float(excellent),
        "marginal_threshold": float(marginal),
    }


__all__ = ["AssayQCDecision", "assay_qc_decision", "classify_z_prime"]
