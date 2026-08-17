"""Public OptiCell package API."""

from qc_pipeline import QCThresholds, analyze_folder, analyze_image, analyze_paths, extract_object_features
from quantitative import add_spatial_features, object_channel_intensity, summarize_spatial_features
from validation import benchmark_segmentation, paired_segmentation_metrics
from .batch import BatchConfig, analyze_paths_parallel
from .statistics import benjamini_hochberg, compare_two_groups, summarize_by_replicate
from .volumetric import nearest_neighbor_distances_3d, summarize_volume, volume_features
from .segmentation import BaseSegmenter, CellposeBackend, ThresholdSegmenter, available_backends, compare_backends, get_backend, register_backend
from .volumetric_segmentation import VolumetricSegmentationResult, segment_threshold_3d
from .tracking3d import Tracking3DConfig, link_frames_3d, summarize_tracks_3d
from .provenance import build_manifest, collect_input_manifest, file_sha256, write_manifest
from .reporting import RuntimeStats, build_report, dataframe_summary, runtime_stats, write_report
from .benchmarking import aggregate_backend_benchmarks, benchmark_backends
from .runtime import RuntimeCapabilities, capabilities, capabilities_dict, measure, preferred_accelerator
from .tracking_events import classify_divisions, detect_time_series_events, detect_transition_events
from .experiment_stats import bootstrap_ci, replicate_effect_summary, summarize_experiment

__all__ = [
    "QCThresholds", "analyze_folder", "analyze_image", "analyze_paths", "extract_object_features",
    "add_spatial_features", "object_channel_intensity", "summarize_spatial_features",
    "benchmark_segmentation", "paired_segmentation_metrics",
    "BatchConfig", "analyze_paths_parallel", "summarize_by_replicate", "compare_two_groups", "benjamini_hochberg",
    "volume_features", "summarize_volume", "nearest_neighbor_distances_3d",
    "BaseSegmenter", "ThresholdSegmenter", "CellposeBackend", "register_backend", "available_backends", "get_backend", "compare_backends",
    "VolumetricSegmentationResult", "segment_threshold_3d",
    "Tracking3DConfig", "link_frames_3d", "summarize_tracks_3d",
    "file_sha256", "collect_input_manifest", "build_manifest", "write_manifest",
    "RuntimeStats", "runtime_stats", "dataframe_summary", "build_report", "write_report",
    "benchmark_backends", "aggregate_backend_benchmarks",
    "RuntimeCapabilities", "capabilities", "capabilities_dict", "preferred_accelerator", "measure",
    "detect_transition_events", "detect_time_series_events", "classify_divisions",
    "bootstrap_ci", "replicate_effect_summary", "summarize_experiment",
]

__version__ = "2.9.0"
