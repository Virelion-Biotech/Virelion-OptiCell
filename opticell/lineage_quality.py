"""Quality diagnostics for cell-tracking and lineage tables."""
from __future__ import annotations

import pandas as pd


def lineage_quality_summary(tracks: pd.DataFrame) -> dict[str, float | int]:
    """Summarize track continuity without assuming biological lineage correctness."""
    required = {"track_id", "frame"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks missing required columns: {sorted(missing)}")
    if tracks.empty:
        return {"n_tracks": 0, "n_observations": 0, "fragmented_tracks": 0, "mean_track_length": 0.0, "track_completeness": 0.0}
    rows = []
    for track_id, group in tracks.groupby("track_id", sort=True):
        frames = sorted(pd.to_numeric(group["frame"], errors="raise").astype(int).unique())
        span = frames[-1] - frames[0] + 1
        rows.append((int(track_id), len(frames), span, len(frames) < span))
    lengths = pd.Series([row[1] for row in rows], dtype=float)
    spans = pd.Series([row[2] for row in rows], dtype=float)
    fragmented = int(sum(row[3] for row in rows))
    return {
        "n_tracks": len(rows),
        "n_observations": int(len(tracks)),
        "fragmented_tracks": fragmented,
        "mean_track_length": float(lengths.mean()),
        "track_completeness": float(lengths.sum() / spans.sum()) if spans.sum() else 0.0,
    }


__all__ = ["lineage_quality_summary"]
