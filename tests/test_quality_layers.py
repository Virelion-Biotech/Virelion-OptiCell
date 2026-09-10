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


def test_artifact_metrics_do_not_treat_float_extrema_as_detector_clipping():
    image = np.full((20, 20), 50.0, dtype=np.float32)
    image[10, 10] = 100.0
    metrics = acquisition_artifact_metrics(image)
    assert metrics["low_clip_fraction"] == 0.0
    assert metrics["high_clip_fraction"] == 0.0
    assert metrics["hot_pixel_fraction"] > 0


def test_explicit_float_intensity_range_enables_clipping_metrics():
    image = np.full((20, 20), 0.5, dtype=np.float32)
    image[:, 0] = 1.0
    metrics = acquisition_artifact_metrics(image, intensity_range=(0.0, 1.0))
    assert metrics["high_clip_fraction"] > 0


def test_bright_structure_is_not_counted_as_hot_pixels():
    image = np.full((25, 25), 100, dtype=np.uint8)
    image[8:17, 8:17] = 255
    metrics = acquisition_artifact_metrics(image)
    assert metrics["hot_pixel_fraction"] == 0.0


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
