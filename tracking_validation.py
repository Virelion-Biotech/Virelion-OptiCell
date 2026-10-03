"""Reference-based tracking quality diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def track_fragmentation(tracks: pd.DataFrame, *, track_column: str = "track_id", object_column: str = "object_id") -> dict[str, float]:
    """Summarize object identity fragmentation when an object has multiple track IDs."""
    required = {track_column, object_column}
    if not required.issubset(tracks.columns):
        raise ValueError(f"missing required columns: {sorted(required - set(tracks.columns))}")
    frame = tracks[[track_column, object_column]].dropna()
    if frame.empty:
        return {"objects": 0.0, "fragmented_objects": 0.0, "fragmentation_rate": np.nan}
    counts = frame.groupby(object_column)[track_column].nunique()
    fragmented = int((counts > 1).sum())
    return {"objects": float(len(counts)), "fragmented_objects": float(fragmented), "fragmentation_rate": fragmented / len(counts)}


def track_gap_rate(tracks: pd.DataFrame, *, track_column: str = "track_id", frame_column: str = "frame") -> dict[str, float]:
    """Measure missing-frame gaps within each track, without inferring biology from gaps."""
    required = {track_column, frame_column}
    if not required.issubset(tracks.columns):
        raise ValueError(f"missing required columns: {sorted(required - set(tracks.columns))}")
    frame = tracks[[track_column, frame_column]].dropna().copy()
    if frame.empty:
        return {"tracks": 0.0, "tracks_with_gaps": 0.0, "gap_rate": np.nan, "gap_count": 0.0}
    gap_count = 0
    tracks_with_gaps = 0
    for _, group in frame.groupby(track_column):
        frames = np.sort(pd.to_numeric(group[frame_column], errors="coerce").dropna().unique())
        if len(frames) > 1:
            gaps = np.diff(frames) - 1
            count = int((gaps > 0).sum())
            gap_count += count
            tracks_with_gaps += int(count > 0)
    total = frame[track_column].nunique()
    return {"tracks": float(total), "tracks_with_gaps": float(tracks_with_gaps), "gap_rate": tracks_with_gaps / total, "gap_count": float(gap_count)}


def track_purity(reference: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Return dominant-label purity for predicted track assignments."""
    ref = pd.Series(reference)
    pred = pd.Series(predicted)
    if len(ref) != len(pred):
        raise ValueError("reference and predicted must have equal length")
    valid = ref.notna() & pred.notna()
    if not valid.any():
        return float("nan")
    table = pd.crosstab(pred[valid], ref[valid])
    return float(table.max(axis=1).sum() / table.to_numpy().sum())


__all__ = ["track_fragmentation", "track_gap_rate", "track_purity"]
