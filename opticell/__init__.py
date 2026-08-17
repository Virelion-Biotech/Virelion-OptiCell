"""Public OptiCell package API."""

from qc_pipeline import QCThresholds, analyze_folder, analyze_image, analyze_paths, extract_object_features
from quantitative import add_spatial_features, object_channel_intensity, summarize_spatial_features
from validation import benchmark_segmentation, paired_segmentation_metrics
from .acceptance import SegmentationAcceptance, segmentation_acceptance
from .artifact_quality import acquisition_artifact_metrics, artifact_burden_score
from .batch import BatchConfig, analyze_paths_parallel
from .benchmarking import aggregate_backend_benchmarks, benchmark_backends
from .exports import dataframe_to_long_form, write_dataframe
from .experiment_qc import normalize_to_controls, plate_edge_effect, plate_qc_summary, robust_zscore
from .experiment_stats import bootstrap_ci, replicate_effect_summary, summarize_experiment
from .lineage import build_lineage_table, summarize_lineages
from .lineage_events import division_consistency, lineage_event_summary
from .lineage_quality import lineage_quality_summary
from .ome_io import OMEImageInfo, load_ome_series, read_ome_info
from .power import two_group_sample_size
from .provenance import build_manifest, collect_input_manifest, file_sha256, write_manifest
from .profiling import ProfileRecord, profile_call, profile_records, summarize_profile
from .reporting import RuntimeStats, build_report, dataframe_summary, runtime_stats, write_report
from .robustness import stable_parameter_subset, summarize_sensitivity
from .runtime import RuntimeCapabilities, capabilities, capabilities_dict, measure, preferred_accelerator
from .screening import percent_control, z_prime_factor
from .screening_qc import AssayQCDecision, assay_qc_decision, classify_z_prime
from .segmentation import BaseSegmenter, CellposeBackend, ThresholdSegmenter, available_backends, compare_backends, get_backend, register_backend
from .sensitivity import threshold_sensitivity
from .statistics import benjamini_hochberg, compare_two_groups, summarize_by_replicate
from .stream_io import iter_array_chunks, iter_tiff_frames, memmap_tiff
from .tracking3d import Tracking3DConfig, link_frames_3d, summarize_tracks_3d
from .tracking_events import classify_divisions, detect_time_series_events, detect_transition_events
from .volumetric import nearest_neighbor_distances_3d, summarize_volume, volume_features
from .volumetric_segmentation import VolumetricSegmentationResult, segment_threshold_3d

__all__ = [
    "AssayQCDecision", "BaseSegmenter", "BatchConfig", "CellposeBackend", "OMEImageInfo", "ProfileRecord", "QCThresholds", "RuntimeCapabilities", "RuntimeStats", "SegmentationAcceptance",
    "ThresholdSegmenter", "Tracking3DConfig", "VolumetricSegmentationResult", "acquisition_artifact_metrics", "add_spatial_features", "aggregate_backend_benchmarks", "analyze_folder",
    "analyze_image", "analyze_paths", "analyze_paths_parallel", "artifact_burden_score", "assay_qc_decision", "available_backends", "benchmark_backends", "benchmark_segmentation",
    "benjamini_hochberg", "bootstrap_ci", "build_lineage_table", "build_manifest", "build_report", "capabilities", "capabilities_dict", "classify_divisions", "classify_z_prime",
    "collect_input_manifest", "compare_backends", "compare_two_groups", "dataframe_summary", "dataframe_to_long_form", "detect_time_series_events", "detect_transition_events",
    "division_consistency", "extract_object_features", "file_sha256", "get_backend", "iter_array_chunks", "iter_tiff_frames", "lineage_event_summary", "lineage_quality_summary",
    "link_frames_3d", "load_ome_series", "measure", "memmap_tiff", "nearest_neighbor_distances_3d", "normalize_to_controls", "object_channel_intensity", "paired_segmentation_metrics",
    "percent_control", "plate_edge_effect", "plate_qc_summary", "preferred_accelerator", "profile_call", "profile_records", "read_ome_info", "register_backend",
    "replicate_effect_summary", "robust_zscore", "runtime_stats", "segment_threshold_3d", "segmentation_acceptance", "stable_parameter_subset", "summarize_by_replicate",
    "summarize_experiment", "summarize_lineages", "summarize_profile", "summarize_sensitivity", "summarize_spatial_features", "summarize_tracks_3d", "summarize_volume",
    "threshold_sensitivity", "two_group_sample_size", "volume_features", "write_dataframe", "write_manifest", "write_report", "z_prime_factor",
]

__version__ = "2.14.0"
