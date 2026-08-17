import numpy as np
import pandas as pd

from statistics import benjamini_hochberg, compare_two_groups, permutation_pvalue, summarize_by_replicate
from texture import basic_texture_features, object_texture_features


def test_replicate_aggregation_and_group_comparison():
    frame = pd.DataFrame({
        "replicate": ["r1", "r1", "r2", "r2", "r3", "r3"],
        "condition": ["A", "A", "A", "A", "B", "B"],
        "area": [10, 12, 11, 13, 20, 21],
    })
    reps = summarize_by_replicate(frame, "replicate", ["area"], ["condition"])
    assert len(reps) == 3
    result = compare_two_groups(reps, "area_mean", "condition", "A", "B", n_permutations=500, seed=1)
    assert result["n_group_a"] == 2
    assert result["n_group_b"] == 1


def test_permutation_and_fdr_are_bounded():
    p = permutation_pvalue([1, 2, 3], [10, 11, 12], n_permutations=500, seed=0)
    assert 0 <= p <= 1
    q = benjamini_hochberg([0.01, 0.04, 0.2, np.nan])
    assert np.all((q[:3] >= 0) & (q[:3] <= 1))
    assert np.isnan(q[3])


def test_texture_features_return_finite_values():
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 180
    labels = np.zeros_like(image, dtype=np.int32)
    labels[8:24, 8:24] = 1
    summary = basic_texture_features(image)
    object_df = object_texture_features(image, labels)
    assert np.isfinite(summary["intensity_entropy"])
    assert len(object_df) == 1
    assert np.isfinite(object_df["texture_gradient_mean"].iloc[0])
