"""Auditable lineage graph construction from tracking/event tables."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _strict_integer(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    malformed = series.notna() & numeric.isna()
    if malformed.any():
        examples = series.loc[malformed].astype(str).head(3).tolist()
        raise ValueError(f"{name} contains non-numeric values: {examples}")
    finite = numeric.to_numpy(dtype=float)
    if not np.isfinite(finite).all():
        raise ValueError(f"{name} must contain finite numeric values")
    if not np.equal(finite, np.floor(finite)).all():
        raise ValueError(f"{name} must contain integer values")
    return numeric.astype(int)


def build_lineage_table(tracks: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach parent track IDs to child observations from validated split events."""
    required = {"track_id", "frame", "label"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks missing required columns: {sorted(missing)}")
    nodes = tracks[["track_id", "frame", "label"]].copy()
    nodes["track_id"] = _strict_integer(nodes["track_id"], "track_id")
    nodes["frame"] = _strict_integer(nodes["frame"], "frame")
    nodes["label"] = _strict_integer(nodes["label"], "label")
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
        frame_value = event.get("frame")
        frame_numeric = pd.to_numeric(pd.Series([frame_value]), errors="coerce").iloc[0]
        if pd.isna(frame_numeric) or not np.isfinite(float(frame_numeric)) or float(frame_numeric) != int(frame_numeric) or int(frame_numeric) < 1:
            raise ValueError("split event frame must be a positive integer")
        frame = int(frame_numeric)

        parent_label_value = event.get("parent_label")
        parent_numeric = pd.to_numeric(pd.Series([parent_label_value]), errors="coerce").iloc[0]
        if pd.isna(parent_numeric) or not np.isfinite(float(parent_numeric)) or float(parent_numeric) != int(parent_numeric) or int(parent_numeric) <= 0:
            raise ValueError("split event parent_label must be a positive integer")
        parent_label = int(parent_numeric)

        children = event.get("child_labels", ())
        if not isinstance(children, Iterable) or isinstance(children, (str, bytes)):
            children = (children,)
        children = tuple(children)
        if not children:
            raise ValueError("split event child_labels must contain at least one child")

        parent_rows = nodes[(nodes["frame"] == frame - 1) & (nodes["label"] == parent_label)]
        if parent_rows.empty:
            raise ValueError(f"split event parent label {parent_label} is absent at frame {frame - 1}")
        if len(parent_rows) > 1:
            raise ValueError("split event maps to multiple parent track observations")
        parent_track = int(parent_rows.iloc[0]["track_id"])

        seen_children: set[int] = set()
        for child in children:
            child_value = pd.to_numeric(pd.Series([child]), errors="coerce").iloc[0]
            if pd.isna(child_value) or not np.isfinite(float(child_value)) or float(child_value) != int(child_value) or int(child_value) <= 0:
                raise ValueError("split event child_labels must contain positive integers")
            child_label = int(child_value)
            if child_label in seen_children:
                raise ValueError("split event child_labels must be unique")
            seen_children.add(child_label)
            child_rows = nodes[(nodes["frame"] == frame) & (nodes["label"] == child_label)]
            if child_rows.empty:
                raise ValueError(f"split event child label {child_label} is absent at frame {frame}")
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
    frame["track_id"] = _strict_integer(frame["track_id"], "track_id")
    frame["frame"] = _strict_integer(frame["frame"], "frame")
    if (frame[["track_id"]] < 0).any().any() or (frame["frame"] < 0).any():
        raise ValueError("track_id and frame must be non-negative")
    if frame.duplicated(["track_id", "frame"]).any():
        raise ValueError("lineage must contain at most one observation per track_id and frame")
    if "parent_track_id" in frame.columns:
        parent_numeric = pd.to_numeric(frame["parent_track_id"], errors="coerce")
        malformed_parent = frame["parent_track_id"].notna() & parent_numeric.isna()
        if malformed_parent.any():
            examples = frame.loc[malformed_parent, "parent_track_id"].astype(str).head(3).tolist()
            raise ValueError(f"parent_track_id contains non-numeric values: {examples}")
        finite_parent = parent_numeric.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite_parent).all() or not np.equal(finite_parent, np.floor(finite_parent)).all() or (finite_parent < 0).any():
            raise ValueError("parent_track_id must contain non-negative integer values")
        frame["parent_track_id"] = parent_numeric.astype("Int64")
        existing_track_ids = set(frame["track_id"].astype(int))
        referenced_parent_ids = {int(x) for x in frame["parent_track_id"].dropna()}
        missing_parents = sorted(referenced_parent_ids - existing_track_ids)
        if missing_parents:
            raise ValueError(f"parent_track_id references unknown tracks: {missing_parents}")

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
            "parent_track_ids": tuple(sorted({int(x) for x in g["parent_track_id"].dropna()}))
            if "parent_track_id" in g
            else (),
        })
    return pd.DataFrame(rows)
