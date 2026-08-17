from datetime import datetime, timezone

import numpy as np
import pandas as pd

from provenance import build_manifest, collect_input_manifest, write_manifest
from reporting import build_report, dataframe_summary, runtime_stats, write_report
from tracking3d import Tracking3DConfig, link_frames_3d, summarize_tracks_3d


def _single_object(z: int, y: int, x: int, label: int = 1) -> np.ndarray:
    volume = np.zeros((20, 20, 20), dtype=np.int32)
    volume[z:z + 2, y:y + 2, x:x + 2] = label
    return volume


def test_tracking3d_uses_physical_units_and_persists_track():
    frames = [_single_object(2, 3, 4), _single_object(3, 4, 5), _single_object(4, 5, 6)]
    tracks = link_frames_3d(frames, voxel_size=(2.0, 1.0, 0.5), frame_interval=2.0)
    assert tracks["track_id"].nunique() == 1
    assert np.isclose(tracks.iloc[1]["distance_um"], np.sqrt(2.0**2 + 1.0**2 + 0.5**2), atol=1e-6)
    assert tracks.iloc[1]["match_confidence"] > 0.9


def test_tracking3d_supports_short_gaps():
    frames = [_single_object(2, 2, 2), np.zeros((20, 20, 20), dtype=np.int32), _single_object(4, 4, 4)]
    tracks = link_frames_3d(frames, config=Tracking3DConfig(max_distance_um=20, max_gap=1))
    assert tracks["track_id"].nunique() == 1
    assert int(tracks.iloc[-1]["gap"]) == 1


def test_tracking3d_summary_reports_straightness():
    frames = [_single_object(1, 1, 1), _single_object(2, 1, 1), _single_object(3, 1, 1)]
    tracks = link_frames_3d(frames)
    summary = summarize_tracks_3d(tracks)
    assert len(summary) == 1
    assert np.isclose(summary.iloc[0]["straightness"], 1.0)
    assert summary.iloc[0]["path_length_um"] > 0


def test_tracking3d_rejects_invalid_configuration():
    try:
        Tracking3DConfig(max_distance_um=0).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid configuration should fail")


def test_provenance_manifest_hashes_existing_file(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("opticell")
    records = collect_input_manifest([str(source), str(tmp_path / "missing.txt")])
    assert records[0]["exists"] is True
    assert len(records[0]["sha256"]) == 64
    assert records[1]["exists"] is False
    manifest = build_manifest(opticell_version="2.8.0", inputs=[str(source)], parameters={"workers": 4})
    assert manifest["parameters"]["workers"] == 4
    assert manifest["inputs"][0]["sha256"] == records[0]["sha256"]
    out = write_manifest(manifest, str(tmp_path / "manifest.json"))
    assert (tmp_path / "manifest.json").exists()
    assert out.endswith("manifest.json")


def test_reporting_summarizes_results_and_normalizes_nonfinite(tmp_path):
    df = pd.DataFrame({"cells": [10, 20, np.nan], "quality": [80.0, 90.0, 100.0]})
    summary = dataframe_summary(df)
    assert summary["rows"] == 3
    assert summary["numeric"]["cells"]["median"] == 15.0

    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    finished = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
    runtime = runtime_stats("batch", started, finished, 10)
    assert runtime.elapsed_seconds == 2.0
    assert runtime.items_per_second == 5.0

    report = build_report(operation="batch", runtime=runtime, results=df, validation={"score": np.nan})
    path = write_report(report, str(tmp_path / "report.json"))
    assert path.endswith("report.json")
    text = (tmp_path / "report.json").read_text()
    assert '"validation"' in text
    assert "NaN" not in text
