from experiment_audit import audit_experiment


def test_experiment_audit_passes_with_matching_reference():
    audit = audit_experiment(
        parameters={"threshold": 0.5},
        input_hashes={"image": "abc"},
        artifact_score=95,
        segmentation_score=90,
        artifact_status="PASS",
        segmentation_status="PASS",
        reference_manifest={"inputs": {"image": {"sha256": "abc"}}, "parameters": {"threshold": 0.5}},
        candidate_manifest={"inputs": {"image": {"sha256": "abc"}}, "parameters": {"threshold": 0.5}},
    )
    assert audit.status == "PASS"
    assert audit.inputs_match is True
    assert audit.parameters_match is True
    assert audit.fingerprint


def test_experiment_audit_preserves_qc_failure_and_manifest_difference():
    audit = audit_experiment(
        parameters={"threshold": 0.7},
        input_hashes={"image": "new"},
        artifact_score=92,
        segmentation_score=88,
        artifact_status="PASS",
        segmentation_status="FAIL",
        reference_manifest={"inputs": {"image": {"sha256": "old"}}, "parameters": {"threshold": 0.5}},
        candidate_manifest={"inputs": {"image": {"sha256": "new"}}, "parameters": {"threshold": 0.7}},
    )
    assert audit.status == "FAIL"
    assert audit.inputs_match is False
    assert audit.parameters_match is False
    assert any("segmentation QC failed" in reason for reason in audit.qc_reasons)
    assert any("input manifest differs" in reason for reason in audit.qc_reasons)
