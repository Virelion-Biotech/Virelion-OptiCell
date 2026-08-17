import numpy as np
import pandas as pd

from lineage import build_lineage_table, summarize_lineages
from screening import normalize_to_controls, percent_control, plate_edge_effect, robust_zscore, z_prime_factor
from stream_io import iter_array_chunks


def test_lineage_builds_parent_child_relationships():
    tracks = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 3],
            "frame": [0, 1, 2, 2],
            "label": [1, 1, 2, 3],
        }
    )
    events = pd.DataFrame(
        [
            {
                "frame": 2,
                "event": "split",
                "parent_label": 1,
                "child_labels": (2, 3),
            }
        ]
    )
    lineage = build_lineage_table(tracks, events)
    children = lineage.loc[lineage["event"] == "division_child"]
    assert set(children["track_id"]) == {2, 3}
    assert set(children["parent_track_id"].astype(int)) == {1}
    summary = summarize_lineages(lineage)
    assert summary.loc[summary["track_id"] == 2, "is_division_child"].iloc[0]


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


def test_plate_edge_effect_identifies_edge_and_interior():
    frame = pd.DataFrame(
        {
            "well": ["A01", "A02", "B02", "B03", "H12"],
            "signal": [10.0, 11.0, 20.0, 19.0, 9.0],
        }
    )
    result = plate_edge_effect(frame, "signal")
    assert result["edge_median"] == 10.5
    assert result["interior_median"] == 19.5
    assert result["edge_to_interior_ratio"] < 1.0


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
