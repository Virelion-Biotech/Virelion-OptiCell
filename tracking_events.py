"""Explainable split/merge/division/disappearance events for time-lapse masks."""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def _overlap_map(previous: np.ndarray, current: np.ndarray, min_overlap: float) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    prev = np.asarray(previous)
    curr = np.asarray(current)
    if prev.shape != curr.shape:
        raise ValueError("consecutive label images must have identical shapes")
    if prev.ndim != 2 or not np.issubdtype(prev.dtype, np.integer) or not np.issubdtype(curr.dtype, np.integer):
        raise ValueError("consecutive label images must be 2-D integer arrays")
    prev_to_curr: dict[int, set[int]] = {}
    curr_to_prev: dict[int, set[int]] = {}
    for pid in np.unique(prev):
        if pid <= 0:
            continue
        mask = prev == pid
        total = int(mask.sum())
        if total == 0:
            continue
        ids, counts = np.unique(curr[mask], return_counts=True)
        hits = {int(cid) for cid, count in zip(ids, counts) if cid > 0 and count / total >= min_overlap}
        prev_to_curr[int(pid)] = hits
    for cid in np.unique(curr):
        if cid <= 0:
            continue
        mask = curr == cid
        total = int(mask.sum())
        ids, counts = np.unique(prev[mask], return_counts=True)
        hits = {int(pid) for pid, count in zip(ids, counts) if pid > 0 and count / total >= min_overlap}
        curr_to_prev[int(cid)] = hits
    return prev_to_curr, curr_to_prev


def detect_transition_events(previous: np.ndarray, current: np.ndarray, *, frame: int = 1, min_overlap: float = 0.2) -> pd.DataFrame:
    """Detect split, merge, appearance and disappearance events from mask overlap."""
    if not isinstance(frame, (int, np.integer)) or isinstance(frame, bool) or frame < 0:
        raise ValueError("frame must be a non-negative integer")
    if not np.isfinite(min_overlap) or not 0 < min_overlap <= 1:
        raise ValueError("min_overlap must be finite and in (0, 1]")
    p2c, c2p = _overlap_map(previous, current, min_overlap)
    events: list[dict[str, object]] = []
    for pid, children in p2c.items():
        if len(children) >= 2:
            events.append({"frame": frame, "event": "split", "parent_label": pid, "child_labels": tuple(sorted(children)), "degree": len(children)})
    for cid, parents in c2p.items():
        if len(parents) >= 2:
            events.append({"frame": frame, "event": "merge", "parent_labels": tuple(sorted(parents)), "child_label": cid, "degree": len(parents)})
    previous_ids = {int(i) for i in np.unique(previous) if i > 0}
    current_ids = {int(i) for i in np.unique(current) if i > 0}
    for cid in sorted(current_ids - set(c2p)):
        events.append({"frame": frame, "event": "appearance", "child_label": cid, "degree": 0})
    for pid in sorted(previous_ids - set(p2c)):
        events.append({"frame": frame, "event": "disappearance", "parent_label": pid, "degree": 0})
    return pd.DataFrame(events)


def detect_time_series_events(labels_by_time: Sequence[np.ndarray], *, min_overlap: float = 0.2) -> pd.DataFrame:
    """Run transition-event detection over a complete, shape-consistent sequence."""
    labels = list(labels_by_time)
    if not np.isfinite(min_overlap) or not 0 < min_overlap <= 1:
        raise ValueError("min_overlap must be finite and in (0, 1]")
    if len(labels) < 2:
        return pd.DataFrame()
    return pd.concat(
        [detect_transition_events(a, b, frame=i + 1, min_overlap=min_overlap) for i, (a, b) in enumerate(zip(labels[:-1], labels[1:]))],
        ignore_index=True,
    )


def classify_divisions(events: pd.DataFrame, *, min_children: int = 2) -> pd.DataFrame:
    """Convert split events into an auditable division table."""
    if not isinstance(min_children, (int, np.integer)) or isinstance(min_children, bool) or min_children < 2:
        raise ValueError("min_children must be an integer >= 2")
    if events.empty:
        return pd.DataFrame(columns=["frame", "parent_label", "child_labels", "is_division"])
    required = {"event", "degree", "frame", "parent_label", "child_labels"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events missing required columns: {sorted(missing)}")
    split = events.loc[events["event"] == "split"].copy()
    if split.empty:
        return pd.DataFrame(columns=["frame", "parent_label", "child_labels", "is_division"])
    degrees = pd.to_numeric(split["degree"], errors="coerce")
    if degrees.isna().any():
        raise ValueError("split event degrees must be numeric")
    split["is_division"] = degrees.astype(int) >= min_children
    return split[["frame", "parent_label", "child_labels", "is_division"]].reset_index(drop=True)


__all__ = ["detect_transition_events", "detect_time_series_events", "classify_divisions"]
