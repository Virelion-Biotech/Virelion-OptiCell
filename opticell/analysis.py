"""High-level package namespace for advanced OptiCell analyses."""
from opticell.compartments import assign_nuclei_to_cells, compartment_features, segment_nuclei
from opticell.phenotype import Rule, group_phenotype_summary, marker_positivity, score_cells
from opticell.validation import benchmark_segmentation

__all__ = [
    "assign_nuclei_to_cells",
    "compartment_features",
    "segment_nuclei",
    "Rule",
    "group_phenotype_summary",
    "marker_positivity",
    "score_cells",
    "benchmark_segmentation",
]
