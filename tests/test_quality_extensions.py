import numpy as np
import pandas as pd

from reproducibility import analysis_fingerprint, compare_manifests
from screening_advanced import b_score, plate_uniformity, ssmd
from tracking_validation import track_fragmentation, track_gap_rate, track_purity


def test_advanced_screening_metrics_are_deterministic():
    frame = pd.DataFrame({
        "row": ["A", "A", "B", "B"],
        "column": [1, 2, 1, 2],
        "value": [10.0, 11.0, 20.0, 21.0],
    })
    scored = b_score(frame, "value")
    assert np.isfinite(scored["value_bscore"]).all()
    assert ssmd([1, 2, 3], [5, 6, 7]) > 0
    summary = plate_uniformity(frame["value"])
    assert summary["n"] == 4.0


def test_reproducibility_fingerprint_and_manifest_diff():
    first = analysis_fingerprint({"threshold": 0.5}, input_hashes={"a": "abc"})
    second = analysis_fingerprint({"threshold": 0.5}, input_hashes={"a": "abc"})
    assert first == second
    diff = compare_manifests(
        {"inputs": {"a": {"sha256": "abc"}}, "parameters": {"threshold": 0.5}},
        {"inputs": {"a": {"sha256": "def"}}, "parameters": {"threshold": 0.6}},
    )
    assert not diff["inputs_match"] and not diff["parameters_match"]


def test_tracking_quality_diagnostics():
    tracks = pd.DataFrame({
        "track_id": [1, 1, 1, 2, 2],
        "object_id": [10, 10, 11, 20, 20],
        "frame": [0, 1, 3, 0, 2],
    })
    fragmentation = track_fragmentation(tracks)
    assert fragmentation["fragmented_objects"] == 1.0
    gaps = track_gap_rate(tracks)
    assert gaps["gap_count"] == 2.0
    assert track_purity([1, 1, 2, 2], [1, 1, 2, 2]) == 1.0
