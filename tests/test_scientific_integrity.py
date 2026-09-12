import numpy as np
import pandas as pd
import pytest

from experiment_audit import audit_experiment
from profiling import ProfileRecord, ProfiledOperationError, profile_call
from screening import normalize_to_controls, z_prime_factor


def test_audit_downgrades_unchecked_reproducibility_to_review():
    audit = audit_experiment(
        parameters={"threshold": 0.5},
        input_hashes={"image": "abc"},
        artifact_score=95,
        segmentation_score=95,
        artifact_status="PASS",
        segmentation_status="PASS",
    )
    assert audit.status == "REVIEW"
    assert audit.inputs_match is None
    assert audit.parameters_match is None
    assert "reproducibility comparison not evaluated" in audit.qc_reasons


def test_profile_failure_retains_structured_record():
    def fail():
        raise ValueError("broken")

    with pytest.raises(ProfiledOperationError) as excinfo:
        profile_call("broken_op", fail, items=3)
    record = excinfo.value.record
    assert isinstance(record, ProfileRecord)
    assert record.status == "error"
    assert record.items == 3
    assert "ValueError: broken" in record.error


def test_screening_rejects_malformed_numeric_values():
    frame = pd.DataFrame({"condition": ["control", "control", "treated"], "signal": [10.0, "bad", 15.0]})
    with pytest.raises(ValueError, match="non-numeric"):
        normalize_to_controls(frame, "signal", "condition", "control")


def test_z_prime_rejects_nonfinite_observations():
    with pytest.raises(ValueError, match="finite"):
        z_prime_factor([1.0, np.inf], [10.0, 11.0])
