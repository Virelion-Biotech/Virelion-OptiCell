import numpy as np
import pandas as pd

from benchmarking import aggregate_backend_benchmarks, benchmark_backends
from experiment_qc import normalize_to_controls, plate_edge_effect, plate_qc_summary, robust_zscore
from profiling import ProfileRecord, profile_call, profile_records, summarize_profile
from tracking import TrackingConfig, link_frames
from tracking3d import Tracking3DConfig, link_frames_3d
from opticell.segmentation import ThresholdSegmenter


def test_2d_tracking_is_one_to_one_and_handles_crossing_candidates():
    first = np.zeros((40, 40), dtype=np.int32)
    first[10:14, 5:9] = 1
    first[10:14, 20:24] = 2
    second = np.zeros_like(first)
    second[10:14, 11:15] = 3
    second[10:14, 14:18] = 4
    tracks = link_frames([first, second], TrackingConfig(max_distance_px=10))
    assert tracks["track_id"].nunique() == 2
    assert tracks.groupby("frame")["track_id"].nunique().eq(2).all()
    assert "match_confidence" in tracks.columns


def test_3d_tracking_respects_anisotropic_physical_spacing():
    first = np.zeros((4, 20, 20), dtype=np.int32)
    first[1, 5:8, 5:8] = 1
    second = np.zeros_like(first)
    second[2, 5:8, 6:9] = 2
    tracks = link_frames_3d([first, second], voxel_size=(5.0, 1.0, 1.0), config=Tracking3DConfig(max_distance_um=6))
    assert tracks["track_id"].nunique() == 1
    assert float(tracks.iloc[1]["dz_um"]) == 5.0


def test_control_normalization_uses_only_controls():
    df = pd.DataFrame({"condition": ["control", "control", "treated"], "signal": [10.0, 12.0, 20.0]})
    normalized = normalize_to_controls(df, "signal", group_column="condition", control_value="control")
    assert abs(float(normalized.loc[0, "signal_robust_z"])) > 0
    assert float(normalized.loc[2, "signal_robust_z"]) > float(normalized.loc[0, "signal_robust_z"])


def test_plate_qc_and_edge_effects_are_structured():
    df = pd.DataFrame({
        "well_row": ["A", "B", "C", "D"], "well_col": [1, 2, 3, 4],
        "signal": [20.0, 10.0, 10.0, 10.0], "condition": ["control"] * 4,
    })
    edge = plate_edge_effect(df, "signal", edge_rows=["A"], edge_cols=[1, 12])
    assert edge["edge_median"] > edge["interior_median"]
    summary = plate_qc_summary(df, ["signal"], control_group="condition", control_value="control")
    assert set(["metric", "n", "median", "control_robust_sd"]).issubset(summary.columns)


def test_robust_zscore_constant_reference_is_finite():
    values = robust_zscore([1, 2, 3], [2, 2, 2])
    assert np.isfinite(values).all()


def test_benchmark_metadata_and_failure_accounting():
    image = np.zeros((32, 32), dtype=np.uint8)
    image[10:20, 10:20] = 255
    reference = np.zeros_like(image, dtype=np.int32)
    reference[10:20, 10:20] = 1
    good = ThresholdSegmenter(min_area=10)

    class Broken:
        def segment(self, _image):
            raise ValueError("intentional failure")

    result = benchmark_backends(image, reference, {"good": good, "broken": Broken()}, metadata={"dataset": "synthetic"})
    assert set(result["dataset"]) == {"synthetic"}
    assert result.loc[result["backend"] == "broken", "error"].notna().all()
    summary = aggregate_backend_benchmarks([result])
    assert set(["backend", "failed_runs"]).issubset(summary.columns)


def test_profiling_records_are_machine_readable():
    result, record = profile_call("noop", lambda x: x + 1, 2, items=4)
    assert result == 3
    assert record.items == 4
    assert record.items_per_second > 0
    table = profile_records([record, ProfileRecord("second", 1.0, 2)])
    assert len(table) == 2
    summary = summarize_profile([record])
    assert summary["total_items"] == 4
