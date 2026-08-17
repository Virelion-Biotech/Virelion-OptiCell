import pytest

from quality_gate import experiment_quality_gate


def test_quality_gate_uses_conservative_minimum_score():
    result = experiment_quality_gate(artifact_score=98, segmentation_score=82)
    assert result.status == "REVIEW"
    assert result.score == 82
    assert result.artifact_score == 98
    assert result.segmentation_score == 82


def test_quality_gate_preserves_component_failure():
    result = experiment_quality_gate(
        artifact_score=95,
        segmentation_score=95,
        artifact_status="FAIL",
        segmentation_status="PASS",
    )
    assert result.status == "FAIL"
    assert "acquisition artifact QC failed" in result.reasons


def test_quality_gate_rejects_invalid_scores_and_thresholds():
    with pytest.raises(ValueError):
        experiment_quality_gate(artifact_score=101, segmentation_score=90)
    with pytest.raises(ValueError):
        experiment_quality_gate(artifact_score=90, segmentation_score=90, review_threshold=90, pass_threshold=80)
