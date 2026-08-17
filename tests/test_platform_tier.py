import numpy as np
import pandas as pd

from lineage import build_lineage_table, summarize_lineages
from lineage_events import division_consistency, lineage_event_summary
from screening import normalize_to_controls, percent_control, plate_edge_effect, robust_zscore, z_prime_factor
from screening_qc import assay_qc_decision, classify_z_prime
from sensitivity import threshold_sensitivity
from stream_io import iter_array_chunks
from opticell.lineage_quality import lineage_quality_summary
from opticell.ome_io import load_ome_series, read_ome_info
from opticell.power import two_group_sample_size
from opticell.exports import dataframe_to_long_form


def test_lineage_builds_parent_child_relationships():
    tracks = pd.DataFrame({"track_id": [1, 1, 2, 3], "frame": [0, 1, 2, 2], "label": [1, 1, 2, 3]})
    events = pd.DataFrame([{"frame": 2, "event": "split", "parent_label": 1, "child_labels": (2, 3)}])
    lineage = build_lineage_table(tracks, events)
    children = lineage.loc[lineage["event"] == "division_child"]
    assert set(children["track_id"]) == {2, 3}
    assert set(children["parent_track_id"].astype(int)) == {1}
    summary = summarize_lineages(lineage)
    assert summary.loc[summary["track_id"] == 2, "is_division_child"].iloc[0]


def test_lineage_quality_reports_track_fragmentation():
    tracks = pd.DataFrame({"track_id": [1, 1, 2, 2, 2], "frame": [0, 2, 0, 1, 2], "label": [1, 1, 2, 2, 2]})
    summary = lineage_quality_summary(tracks)
    assert summary["n_tracks"] == 2
    assert summary["fragmented_tracks"] == 1
    assert 0.0 <= summary["track_completeness"] <= 1.0


def test_lineage_event_metrics():
    events = pd.DataFrame({"event": ["split", "split", "merge", "appearance", "disappearance"], "degree": [2, 1, 2, 0, 0]})
    summary = lineage_event_summary(events, n_frames=6)
    assert summary["n_events"] == 5
    assert summary["splits"] == 2
    assert summary["events_per_transition"] == 1.0
    consistency = division_consistency(events)
    assert consistency["split_events"] == 2
    assert consistency["consistent_splits"] == 1
    assert consistency["consistency_rate"] == 0.5


def test_screening_normalization_and_z_prime():
    frame = pd.DataFrame({"condition": ["control", "control", "treated"], "signal": [10.0, 20.0, 15.0]})
    normalized = normalize_to_controls(frame, "signal", "condition", "control")
    assert normalized["signal_normalized"].iloc[2] == 1.0
    pct = percent_control(frame, "signal", "condition", "control")
    assert pct["signal_fraction_control"].iloc[2] == 100.0
    robust = robust_zscore(pd.Series([1.0, 1.0, 1.0]))
    assert np.allclose(robust, 0.0)
    zprime = z_prime_factor([10, 11, 9], [100, 101, 99])
    assert zprime > 0.9


def test_screening_qc_decision():
    assert classify_z_prime(0.7) == "PASS"
    assert classify_z_prime(0.2) == "MARGINAL"
    assert classify_z_prime(-0.1) == "FAIL"
    decision = assay_qc_decision(0.7)
    assert decision["status"] == "PASS"
    assert "Z'" in decision["reason"]


def test_plate_edge_effect_identifies_edge_and_interior():
    frame = pd.DataFrame({"well": ["A01", "A02", "B02", "B03", "H12"], "signal": [10.0, 11.0, 20.0, 19.0, 9.0]})
    result = plate_edge_effect(frame, "signal")
    assert result["edge_median"] == 10.0
    assert result["interior_median"] == 19.5
    assert result["edge_to_interior_ratio"] < 1.0


def test_power_analysis_returns_integer_sample_size():
    result = two_group_sample_size(effect_size=0.8, alpha=0.05, power=0.8)
    assert result["n_per_group"] >= 20
    assert result["alpha"] == 0.05


def test_ome_loader_rejects_invalid_series(tmp_path):
    import tifffile
    path = tmp_path / "image.tif"
    tifffile.imwrite(path, np.zeros((4, 5), dtype=np.uint8))
    info = read_ome_info(str(path))
    assert info.shape == (4, 5)
    try:
        load_ome_series(str(path), series_index=3)
    except IndexError:
        pass
    else:
        raise AssertionError("expected invalid series index")


def test_long_form_export_is_deterministic():
    frame = pd.DataFrame({"sample": ["b", "a"], "cell": [2, 1], "area": [30.0, 20.0]})
    out = dataframe_to_long_form(frame, id_columns=["sample", "cell"])
    assert list(out.columns) == ["sample", "cell", "feature", "value"]
    assert list(out["feature"]) == ["area", "area"]


def test_threshold_sensitivity_is_stable_and_sorted():
    image = np.zeros((4, 4), dtype=np.uint8)
    image[1:3, 1:3] = 200

    def segmenter(data, threshold):
        labels = np.zeros_like(data, dtype=np.int32)
        labels[(data >= threshold) & (data > 0)] = 1
        return labels

    result = threshold_sensitivity(image, [180, 100, 200], segmenter)
    assert list(result["threshold"]) == [100.0, 180.0, 200.0]
    assert (result["object_count"] == 1).all()
    assert result["count_cv"].iloc[0] == 0.0


def test_array_chunk_iteration_preserves_axis_shape():
    array = np.arange(24).reshape(4, 3, 2)
    chunks = list(iter_array_chunks(array, axis=0, chunk_size=3))
    assert [chunk.shape for chunk in chunks] == [(3, 3, 2), (1, 3, 2)]
    assert np.array_equal(np.concatenate(chunks, axis=0), array)


def test_stream_validation_rejects_invalid_chunks():
    try:
        list(iter_array_chunks(np.zeros((2, 2)), chunk_size=0))
    except ValueError as exc:
        assert "chunk_size" in str(exc)
    else:
        raise AssertionError("expected invalid chunk size to fail")
