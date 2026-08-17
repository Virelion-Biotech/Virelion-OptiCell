import numpy as np
import pandas as pd
import pytest

from opticell.acceptance import segmentation_acceptance
from opticell.artifact_quality import acquisition_artifact_metrics, artifact_burden_score
from opticell.robustness import stable_parameter_subset, summarize_sensitivity


def test_acquisition_artifact_metrics_and_score():
    image = np.full((20, 20), 100, dtype=np.uint8)
    image[:, -1] = 255
    metrics = acquisition_artifact_metrics(image)
    assert metrics["high_clip_fraction"] > 0
    score = artifact_burden_score(metrics)
    assert 0 <= score <= 100


def test_segmentation_acceptance_pass_review_fail():
    passed = segmentation_acceptance(quality_score=90, border_fraction=0.05, tiny_object_fraction=0.05, merged_object_fraction=0.02)
    assert passed.status == "PASS"
    review = segmentation_acceptance(quality_score=75, border_fraction=0.40, tiny_object_fraction=0.05, merged_object_fraction=0.02)
    assert review.status == "REVIEW"
    failed = segmentation_acceptance(quality_score=40, border_fraction=0.50, tiny_object_fraction=0.60, merged_object_fraction=0.30)
    assert failed.status == "FAIL"
    with pytest.raises(ValueError):
        segmentation_acceptance(quality_score=float("nan"))


def test_robustness_summary_and_subset():
    table = pd.DataFrame({"threshold": [80, 100, 120], "object_count": [100, 102, 98]})
    summary = summarize_sensitivity(table)
    assert summary["n_settings"] == 3
    assert summary["stability"] == "HIGH"
    subset = stable_parameter_subset(table, max_cv=0.03)
    assert len(subset) == 3
    assert "relative_deviation" in subset.columns
    with pytest.raises(ValueError):
        stable_parameter_subset(table, max_cv=-0.1)
