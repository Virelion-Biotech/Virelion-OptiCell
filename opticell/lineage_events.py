"""Lineage consistency and event-rate diagnostics."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def lineage_event_summary(events: pd.DataFrame, *, n_frames: int | None = None) -> dict[str, float | int]:
    """Summarize split/merge/appearance/disappearance event burden.

    An empty event table is a valid observation of zero recorded events; it is
    not evidence that the lineage model is perfectly consistent.
    """
    if n_frames is not None and (not isinstance(n_frames, (int, np.integer)) or isinstance(n_frames, bool) or n_frames < 0):
        raise ValueError("n_frames must be a non-negative integer when provided")
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
    """Check how many observed split events satisfy a minimum child count.

    If no split events are observed, ``consistency_rate`` is NaN because the
    statistic is not estimable rather than 100% consistent.
    """
    if not isinstance(min_children, (int, np.integer)) or isinstance(min_children, bool) or min_children < 2:
        raise ValueError("min_children must be an integer >= 2")
    if events is None or events.empty:
        return {"split_events": 0, "consistent_splits": 0, "consistency_rate": math.nan}
    if "event" not in events.columns:
        raise ValueError("events must contain an 'event' column")
    split = events.loc[events["event"].astype(str) == "split"].copy()
    if split.empty:
        return {"split_events": 0, "consistent_splits": 0, "consistency_rate": math.nan}
    if "degree" in split:
        degrees = pd.to_numeric(split["degree"], errors="coerce")
        malformed = split["degree"].notna() & degrees.isna()
        if malformed.any():
            examples = split.loc[malformed, "degree"].astype(str).head(3).tolist()
            raise ValueError(f"split event degree contains non-numeric values: {examples}")
        if not np.isfinite(degrees.dropna().to_numpy(float)).all() or not np.equal(degrees.dropna().to_numpy(float), np.floor(degrees.dropna().to_numpy(float))).all():
            raise ValueError("split event degree must contain finite integer values")
        consistent = int((degrees >= min_children).fillna(False).sum())
    elif "child_labels" in split:
        consistent = 0
        for children in split["child_labels"]:
            if isinstance(children, (str, bytes)) or children is None:
                raise ValueError("split event child_labels must be iterable, not a string/null")
            try:
                child_values = list(children)
            except TypeError as exc:
                raise ValueError("split event child_labels must be iterable") from exc
            if len(child_values) < 1:
                raise ValueError("split event child_labels must not be empty")
            consistent += len(child_values) >= min_children
    else:
        raise ValueError("split events need 'degree' or 'child_labels'")
    total = int(len(split))
    return {"split_events": total, "consistent_splits": consistent, "consistency_rate": float(consistent / total)}


__all__ = ["division_consistency", "lineage_event_summary"]
