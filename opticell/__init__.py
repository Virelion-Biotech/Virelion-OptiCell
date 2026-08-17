"""Public OptiCell package API.

Legacy top-level modules remain available for backward compatibility while the
stable public namespace lives under ``opticell``.
"""

from qc_pipeline import QCThresholds, analyze_folder, analyze_image, analyze_paths, extract_object_features
from quantitative import add_spatial_features, object_channel_intensity, summarize_spatial_features
from validation import benchmark_segmentation
from .batch import BatchConfig, analyze_paths_parallel
from .statistics import benjamini_hochberg, compare_two_groups, summarize_by_replicate
from .volumetric import nearest_neighbor_distances_3d, summarize_volume, volume_features
from .segmentation import BaseSegmenter, CellposeBackend, ThresholdSegmenter, available_backends, compare_backends, get_backend, register_backend
from .volumetric_segmentation import VolumetricSegmentationResult, segment_threshold_3d
from .tracking3d import Tracking3DConfig, link_frames_3d, summarize_tracks_3d
from .provenance import build_manifest, collect_input_manifest, file_sha256, write_manifest
from .reporting import RuntimeStats, build_report, dataframe_summary, runtime_stats, write_report

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
    "BatchConfig",
    "analyze_paths_parallel",
    "summarize_by_replicate",
    "compare_two_groups",
    "benjamini_hochberg",
    "volume_features",
    "summarize_volume",
    "nearest_neighbor_distances_3d",
    "BaseSegmenter",
    "ThresholdSegmenter",
    "CellposeBackend",
    "register_backend",
    "available_backends",
    "get_backend",
    "compare_backends",
    "VolumetricSegmentationResult",
    "segment_threshold_3d",
    "Tracking3DConfig",
    "link_frames_3d",
    "summarize_tracks_3d",
    "file_sha256",
    "collect_input_manifest",
    "build_manifest",
    "write_manifest",
    "RuntimeStats",
    "runtime_stats",
    "dataframe_summary",
    "build_report",
    "write_report",
]

__version__ = "2.8.0"
