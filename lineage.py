"""Auditable lineage graph construction from tracking/event tables."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def build_lineage_table(tracks: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach parent track IDs to child observations from validated split events."""
    required = {"track_id", "frame", "label"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks missing required columns: {sorted(missing)}")
    nodes = tracks[["track_id", "frame", "label"]].copy()
    numeric = nodes[["track_id", "frame", "label"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("track_id, frame, and label must be finite numeric values")
    if not np.equal(numeric.to_numpy(dtype=float), np.floor(numeric.to_numpy(dtype=float))).all():
        raise ValueError("track_id, frame, and label must contain integer values")
    nodes["track_id"] = numeric["track_id"].astype(int)
    nodes["frame"] = numeric["frame"].astype(int)
    nodes["label"] = numeric["label"].astype(int)
    if (nodes[["track_id", "label"]] < 0).any().any() or (nodes["frame"] < 0).any():
        raise ValueError("track_id, frame, and label must be non-negative")
    if nodes.duplicated(["track_id", "frame"]).any():
        raise ValueError("tracks must contain at most one observation per track_id and frame")
    nodes["parent_track_id"] = pd.Series(pd.NA, index=nodes.index, dtype="Int64")
    nodes["event"] = "observation"
    if events is None or events.empty:
        return nodes
    event_required = {"event", "frame"}
    missing_events = event_required - set(events.columns)
    if missing_events:
        raise ValueError(f"events missing required columns: {sorted(missing_events)}")

    for _, event in events.iterrows():
        if event.get("event") != "split":
            continue
        frame_value = pd.to_numeric(pd.Series([event.get("frame")]), errors="coerce").iloc[0]
        if pd.isna(frame_value) or float(frame_value) != int(frame_value) or int(frame_value) < 1:
            raise ValueError("split event frame must be a positive integer")
        frame = int(frame_value)
        parent_label_value = pd.to_numeric(pd.Series([event.get("parent_label")]), errors="coerce").iloc[0]
        if pd.isna(parent_label_value) or float(parent_label_value) != int(parent_label_value) or int(parent_label_value) <= 0:
            raise ValueError("split event parent_label must be a positive integer")
        parent_label = int(parent_label_value)
        children = event.get("child_labels", ())
        if not isinstance(children, Iterable) or isinstance(children, (str, bytes)):
            children = (children,)
        children = tuple(children)
        if not children:
            raise ValueError("split event child_labels must contain at least one child")
        parent_rows = nodes[(nodes["frame"] == frame - 1) & (nodes["label"] == parent_label)]
        if parent_rows.empty:
            continue
        if len(parent_rows) > 1:
            raise ValueError("split event maps to multiple parent track observations")
        parent_track = int(parent_rows.iloc[0]["track_id"])
        for child in children:
            child_value = pd.to_numeric(pd.Series([child]), errors="coerce").iloc[0]
            if pd.isna(child_value) or float(child_value) != int(child_value) or int(child_value) <= 0:
                raise ValueError("split event child_labels must contain positive integers")
            child_rows = nodes[(nodes["frame"] == frame) & (nodes["label"] == int(child_value))]
            if child_rows.empty:
                continue
            if len(child_rows) > 1:
                raise ValueError("split event maps to multiple child track observations")
            idx = child_rows.index[0]
            existing_parent = nodes.loc[idx, "parent_track_id"]
            if pd.notna(existing_parent) and int(existing_parent) != parent_track:
                raise ValueError("child observation is assigned to multiple parent tracks")
            nodes.loc[idx, "parent_track_id"] = parent_track
            nodes.loc[idx, "event"] = "division_child"
    return nodes


def summarize_lineages(lineage: pd.DataFrame) -> pd.DataFrame:
    """Summarize track duration and division-child status."""
    required = {"track_id", "frame"}
    missing = required - set(lineage.columns)
    if missing:
        raise ValueError(f"lineage missing required columns: {sorted(missing)}")
    frame = lineage.copy()
    numeric = frame[["track_id", "frame"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("track_id and frame must contain finite numeric values")
    if not np.equal(numeric.to_numpy(dtype=float), np.floor(numeric.to_numpy(dtype=float))).all():
        raise ValueError("track_id and frame must contain integer values")
    if frame.duplicated(["track_id", "frame"]).any():
        raise ValueError("lineage must contain at most one observation per track_id and frame")
    if "parent_track_id" in frame.columns and (pd.to_numeric(frame["parent_track_id"], errors="coerce") < 0).fillna(False).any():
        raise ValueError("parent_track_id must be non-negative")
    rows = []
    for track_id, group in frame.groupby("track_id", sort=True):
        g = group.sort_values("frame")
        rows.append({
            "track_id": int(track_id),
            "start_frame": int(g["frame"].min()),
            "end_frame": int(g["frame"].max()),
            "frames": int(len(g)),
            "duration_frames": int(g["frame"].max() - g["frame"].min()),
            "is_division_child": bool((g.get("event") == "division_child").any()) if "event" in g else False,
            "parent_track_ids": tuple(sorted({int(x) for x in pd.to_numeric(g["parent_track_id"], errors="coerce").dropna()}))
            if "parent_track_id" in g
            else (),
        })
    return pd.DataFrame(rows)
