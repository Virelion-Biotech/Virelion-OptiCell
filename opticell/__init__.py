"""Public OptiCell package API.

Legacy top-level modules remain available for backward compatibility while the
stable public namespace lives under ``opticell``.
"""

from qc_pipeline import QCThresholds, analyze_folder, analyze_image, analyze_paths, extract_object_features
from quantitative import add_spatial_features, object_channel_intensity, summarize_spatial_features
from validation import benchmark_segmentation
from .statistics import benjamini_hochberg, compare_two_groups, summarize_by_replicate

__all__ = [
    "QCThresholds",
    "analyze_folder",
    "analyze_image",
    "analyze_paths",
    "extract_object_features",
    "add_spatial_features",
    "object_channel_intensity",
    "summarize_spatial_features",
    "benchmark_segmentation",
    "summarize_by_replicate",
    "compare_two_groups",
    "benjamini_hochberg",
]

__version__ = "2.4.0"
