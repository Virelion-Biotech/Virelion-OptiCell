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
    environment_match: bool | None
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
    """Combine QC and reproducibility checks without hiding unevaluated evidence."""
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
    environment_match: bool | None = None
    reasons = list(gate.reasons)
    status = gate.status

    if reference_manifest is None or candidate_manifest is None:
        reasons.append("reproducibility comparison not evaluated")
        if status == "PASS":
            status = "REVIEW"
        return ExperimentAudit(status, gate.score, fingerprint, inputs_match, parameters_match, environment_match, tuple(reasons))

    diff = compare_manifests(reference_manifest, candidate_manifest)
    inputs_match = bool(diff["inputs_match"])
    parameters_match = bool(diff["parameters_match"])
    environment_match = diff.get("environment_match")
    if not inputs_match:
        reasons.append("input manifest differs from reference")
        if diff.get("unverifiable_inputs"):
            reasons.append("input manifest contains entries without SHA-256 digests")
    if not parameters_match:
        reasons.append("analysis parameters differ from reference")
    if environment_match is None:
        reasons.append("runtime environment comparison not evaluated")
    elif not environment_match:
        reasons.append("runtime environment differs from reference")

    if not inputs_match or not parameters_match:
        status = "FAIL"
    elif environment_match is None and status == "PASS":
        status = "REVIEW"
    elif environment_match is False and status == "PASS":
        status = "REVIEW"
    return ExperimentAudit(status, gate.score, fingerprint, inputs_match, parameters_match, environment_match, tuple(reasons))


__all__ = ["ExperimentAudit", "audit_experiment"]
