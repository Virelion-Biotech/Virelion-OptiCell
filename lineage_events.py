"""Lineage consistency and event-rate diagnostics."""
from __future__ import annotations

import pandas as pd


def lineage_event_summary(events: pd.DataFrame, *, n_frames: int | None = None) -> dict[str, float | int]:
    """Summarize split/merge/appearance/disappearance event burden."""
    if events is None or events.empty:
        return {
            "n_events": 0,
            "splits": 0,
            "merges": 0,
            "appearances": 0,
            "disappearances": 0,
            "events_per_transition": 0.0,
        }
    if "event" not in events.columns:
        raise ValueError("events must contain an 'event' column")
    counts = events["event"].astype(str).value_counts()
    total = int(len(events))
    transitions = max(int(n_frames) - 1, 1) if n_frames is not None else 1
    return {
        "n_events": total,
        "splits": int(counts.get("split", 0)),
        "merges": int(counts.get("merge", 0)),
        "appearances": int(counts.get("appearance", 0)),
        "disappearances": int(counts.get("disappearance", 0)),
        "events_per_transition": float(total / transitions),
    }


def division_consistency(events: pd.DataFrame, *, min_children: int = 2) -> dict[str, float | int]:
    """Check how many split events satisfy a minimum child count."""
    if min_children < 2:
        raise ValueError("min_children must be at least 2")
    if events is None or events.empty:
        return {"split_events": 0, "consistent_splits": 0, "consistency_rate": 1.0}
    split = events.loc[events["event"].astype(str) == "split"]
    if split.empty:
        return {"split_events": 0, "consistent_splits": 0, "consistency_rate": 1.0}
    if "degree" in split:
        consistent = int((pd.to_numeric(split["degree"], errors="coerce") >= min_children).sum())
    elif "child_labels" in split:
        consistent = int(sum(len(x) >= min_children for x in split["child_labels"]))
    else:
        raise ValueError("split events need 'degree' or 'child_labels'")
    total = int(len(split))
    return {"split_events": total, "consistent_splits": consistent, "consistency_rate": float(consistent / total)}


__all__ = ["division_consistency", "lineage_event_summary"]
