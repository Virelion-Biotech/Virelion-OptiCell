"""Structured experiment audit reports combining QC and reproducibility evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from quality_gate import ExperimentQualityGate, experiment_quality_gate
from reproducibility import analysis_fingerprint, compare_manifests


@dataclass(frozen=True)
class ExperimentAudit:
    """Machine-readable audit result for an analysis run."""

    status: str
    score: float
    fingerprint: str
    inputs_match: bool | None
    parameters_match: bool | None
    qc_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_experiment(
    *,
    parameters: Mapping[str, Any],
    input_hashes: Mapping[str, str] | None,
    artifact_score: float,
    segmentation_score: float,
    artifact_status: str | None = None,
    segmentation_status: str | None = None,
    reference_manifest: Mapping[str, Any] | None = None,
    candidate_manifest: Mapping[str, Any] | None = None,
    review_threshold: float = 70.0,
    pass_threshold: float = 85.0,
) -> ExperimentAudit:
    """Combine QC and reproducibility checks without hiding component evidence."""
    gate: ExperimentQualityGate = experiment_quality_gate(
        artifact_score=artifact_score,
        segmentation_score=segmentation_score,
        artifact_status=artifact_status,
        segmentation_status=segmentation_status,
        review_threshold=review_threshold,
        pass_threshold=pass_threshold,
    )
    fingerprint = analysis_fingerprint(parameters, input_hashes=input_hashes)
    inputs_match: bool | None = None
    parameters_match: bool | None = None
    reasons = list(gate.reasons)
    if reference_manifest is not None and candidate_manifest is not None:
        diff = compare_manifests(reference_manifest, candidate_manifest)
        inputs_match = bool(diff["inputs_match"])
        parameters_match = bool(diff["parameters_match"])
        if not inputs_match:
            reasons.append("input manifest differs from reference")
        if not parameters_match:
            reasons.append("analysis parameters differ from reference")
    return ExperimentAudit(gate.status, gate.score, fingerprint, inputs_match, parameters_match, tuple(reasons))


__all__ = ["ExperimentAudit", "audit_experiment"]
