"""Auditable lineage graph construction from tracking/event tables."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def build_lineage_table(tracks: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach parent track IDs to child observations from split events."""
    required = {"track_id", "frame", "label"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks missing required columns: {sorted(missing)}")
    nodes = tracks[["track_id", "frame", "label"]].copy()
    nodes["track_id"] = nodes["track_id"].astype(int)
    nodes["frame"] = nodes["frame"].astype(int)
    nodes["parent_track_id"] = pd.Series(pd.NA, index=nodes.index, dtype="Int64")
    nodes["event"] = "observation"
    if events is None or events.empty:
        return nodes
    for _, event in events.iterrows():
        if event.get("event") != "split":
            continue
        frame = int(event.get("frame", 0))
        parent_label = event.get("parent_label")
        children = event.get("child_labels", ())
        parent_rows = nodes[(nodes["frame"] == frame - 1) & (nodes["label"] == parent_label)]
        if parent_rows.empty:
            continue
        parent_track = int(parent_rows.iloc[0]["track_id"])
        if not isinstance(children, Iterable) or isinstance(children, (str, bytes)):
            children = (children,)
        for child in children:
            child_rows = nodes[(nodes["frame"] == frame) & (nodes["label"] == int(child))]
            if child_rows.empty:
                continue
            idx = child_rows.index[0]
            nodes.loc[idx, "parent_track_id"] = parent_track
            nodes.loc[idx, "event"] = "division_child"
    return nodes


def summarize_lineages(lineage: pd.DataFrame) -> pd.DataFrame:
    """Summarize track duration and division-child status."""
    required = {"track_id", "frame"}
    missing = required - set(lineage.columns)
    if missing:
        raise ValueError(f"lineage missing required columns: {sorted(missing)}")
    rows = []
    for track_id, group in lineage.groupby("track_id", sort=True):
        g = group.sort_values("frame")
        rows.append(
            {
                "track_id": int(track_id),
                "start_frame": int(g["frame"].min()),
                "end_frame": int(g["frame"].max()),
                "frames": int(len(g)),
                "duration_frames": int(g["frame"].max() - g["frame"].min()),
                "is_division_child": bool((g.get("event") == "division_child").any()) if "event" in g else False,
                "parent_track_ids": tuple(sorted({int(x) for x in g["parent_track_id"].dropna()}))
                if "parent_track_id" in g
                else (),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["build_lineage_table", "summarize_lineages"]
