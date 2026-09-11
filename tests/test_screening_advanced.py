import numpy as np
import pandas as pd
import pytest

from screening_advanced import b_score, plate_uniformity, ssmd


def test_ssmd_uses_strict_standardization_denominator():
    control = np.array([0.0, 2.0])
    treatment = np.array([2.0, 4.0])
    # Means differ by 2; each sample variance is 2, so SSMD = 2 / sqrt(4) = 1.
    assert np.isclose(ssmd(control, treatment), 1.0)


def test_ssmd_rejects_zero_combined_variance():
    with pytest.raises(ValueError, match="variance is zero"):
        ssmd([1, 1], [2, 2])


def test_b_score_requires_coordinates_for_measured_values():
    frame = pd.DataFrame({"value": [1.0, 2.0], "row": [1, np.nan], "column": [1, 2]})
    with pytest.raises(ValueError, match="coordinates cannot be missing"):
        b_score(frame, "value")


def test_b_score_constant_plate_returns_centered_zero_scores():
    frame = pd.DataFrame({"value": [5.0, 5.0, 5.0, 5.0], "row": [1, 1, 2, 2], "column": [1, 2, 1, 2]})
    out = b_score(frame, "value")
    assert np.allclose(out["value_bscore"], 0.0)


def test_plate_uniformity_rejects_infinite_values():
    with pytest.raises(ValueError, match="finite"):
        plate_uniformity([1.0, np.inf])
