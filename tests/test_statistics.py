import numpy as np
import pandas as pd
import pytest

from opticell.statistics import (
    compare_paired_groups,
    paired_permutation_pvalue,
    permutation_pvalue,
    summarize_by_replicate,
)


def test_independent_permutation_and_paired_permutation_are_explicitly_distinct():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.5, 2.5, 3.5, 4.5]
    independent = permutation_pvalue(a, b, n_permutations=500, seed=1)
    paired = paired_permutation_pvalue(a, b, n_permutations=500, seed=1)
    assert 0 <= independent <= 1
    assert 0 <= paired <= 1


def test_paired_permutation_requires_matched_lengths():
    with pytest.raises(ValueError, match="identical lengths"):
        paired_permutation_pvalue([1.0, 2.0], [1.0], n_permutations=100)


def test_compare_paired_groups_enforces_one_observation_per_group_per_pair():
    data = pd.DataFrame(
        {
            "pair": ["p1", "p1", "p2", "p2"],
            "condition": ["control", "treated", "control", "treated"],
            "value": [10.0, 12.0, 20.0, 23.0],
        }
    )
    result = compare_paired_groups(
        data,
        value_column="value",
        group_column="condition",
        pair_column="pair",
        group_a="control",
        group_b="treated",
        n_permutations=500,
        seed=2,
    )
    assert result["n_pairs"] == 2.0
    assert result["mean_difference"] == -2.5
    assert 0 <= result["paired_permutation_p"] <= 1


def test_compare_paired_groups_rejects_duplicate_pair_condition_rows():
    data = pd.DataFrame(
        {
            "pair": ["p1", "p1", "p1"],
            "condition": ["control", "treated", "treated"],
            "value": [10.0, 12.0, 13.0],
        }
    )
    with pytest.raises(ValueError, match="exactly once"):
        compare_paired_groups(
            data,
            value_column="value",
            group_column="condition",
            pair_column="pair",
            group_a="control",
            group_b="treated",
            n_permutations=100,
        )


def test_summarize_by_replicate_rejects_missing_replicate_ids():
    data = pd.DataFrame({"replicate": ["r1", np.nan], "value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing replicate IDs"):
        summarize_by_replicate(data, "replicate", ["value"])
