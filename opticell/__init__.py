"""Public OptiCell package API."""

from qc_pipeline import QCThresholds, analyze_folder, analyze_image, analyze_paths, extract_object_features
from quantitative import add_spatial_features, object_channel_intensity, summarize_spatial_features
from validation import benchmark_segmentation, paired_segmentation_metrics
from .batch import BatchConfig, analyze_paths_parallel
from .benchmarking import aggregate_backend_benchmarks, benchmark_backends
from .experiment_qc import normalize_to_controls, plate_edge_effect, plate_qc_summary, robust_zscore
from .experiment_stats import bootstrap_ci, replicate_effect_summary, summarize_experiment
from .lineage import build_lineage_table, summarize_lineages
from .provenance import build_manifest, collect_input_manifest, file_sha256, write_manifest
from .profiling import ProfileRecord, profile_call, profile_records, summarize_profile
from .reporting import RuntimeStats, build_report, dataframe_summary, runtime_stats, write_report
from .runtime import RuntimeCapabilities, capabilities, capabilities_dict, measure, preferred_accelerator
from .screening import percent_control, z_prime_factor
from .segmentation import BaseSegmenter, CellposeBackend, ThresholdSegmenter, available_backends, compare_backends, get_backend, register_backend
from .statistics import benjamini_hochberg, compare_two_groups, summarize_by_replicate
from .stream_io import iter_array_chunks, iter_tiff_frames, memmap_tiff
from .tracking3d import Tracking3DConfig, link_frames_3d, summarize_tracks_3d
from .tracking_events import classify_divisions, detect_time_series_events, detect_transition_events
from .volumetric import nearest_neighbor_distances_3d, summarize_volume, volume_features
from .volumetric_segmentation import VolumetricSegmentationResult, segment_threshold_3d

__all__ = [
    "BaseSegmenter", "BatchConfig", "CellposeBackend", "ProfileRecord", "QCThresholds", "RuntimeCapabilities", "RuntimeStats",
    "ThresholdSegmenter", "Tracking3DConfig", "VolumetricSegmentationResult", "add_spatial_features", "aggregate_backend_benchmarks",
    "analyze_folder", "analyze_image", "analyze_paths", "analyze_paths_parallel", "available_backends", "benchmark_backends",
    "benchmark_segmentation", "benjamini_hochberg", "bootstrap_ci", "build_lineage_table", "build_manifest", "build_report",
    "capabilities", "capabilities_dict", "classify_divisions", "collect_input_manifest", "compare_backends", "compare_two_groups",
    "dataframe_summary", "detect_time_series_events", "detect_transition_events", "extract_object_features", "file_sha256", "get_backend",
    "iter_array_chunks", "iter_tiff_frames", "link_frames_3d", "measure", "memmap_tiff", "nearest_neighbor_distances_3d",
    "normalize_to_controls", "object_channel_intensity", "paired_segmentation_metrics", "percent_control", "plate_edge_effect",
    "plate_qc_summary", "preferred_accelerator", "profile_call", "profile_records", "register_backend", "replicate_effect_summary",
    "robust_zscore", "runtime_stats", "segment_threshold_3d", "summarize_by_replicate", "summarize_experiment", "summarize_lineages",
    "summarize_profile", "summarize_spatial_features", "summarize_tracks_3d", "summarize_volume", "volume_features", "write_manifest",
    "write_report", "z_prime_factor",
]

__version__ = "2.11.0"
