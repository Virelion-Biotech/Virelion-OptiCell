"""Public OptiCell package API.

Legacy top-level modules remain available for backward compatibility.
"""

from qc_pipeline import QCThresholds, analyze_folder, analyze_image, analyze_paths, extract_object_features
from quantitative import add_spatial_features, object_channel_intensity, summarize_spatial_features
from validation import benchmark_segmentation

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
]

__version__ = "2.4.0"
