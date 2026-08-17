"""Stable package namespace for replicate-aware experiment statistics."""
from statistics import (
    benjamini_hochberg,
    compare_two_groups,
    effect_size_mean_difference,
    group_summary,
    permutation_pvalue,
    summarize_by_replicate,
)

__all__ = [
    "benjamini_hochberg",
    "compare_two_groups",
    "effect_size_mean_difference",
    "group_summary",
    "permutation_pvalue",
    "summarize_by_replicate",
]
