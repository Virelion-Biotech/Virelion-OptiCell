import pandas as pd
import pytest

from lineage import build_lineage_table, summarize_lineages


def test_build_lineage_rejects_orphan_split_event():
    tracks = pd.DataFrame({"track_id": [1, 2], "frame": [0, 1], "label": [1, 2]})
    events = pd.DataFrame([{"frame": 2, "event": "split", "parent_label": 99, "child_labels": (2, 3)}])
    with pytest.raises(ValueError, match="parent label 99 is absent"):
        build_lineage_table(tracks, events)


def test_summarize_lineages_rejects_malformed_parent_id():
    lineage = pd.DataFrame({
        "track_id": [1, 2],
        "frame": [0, 1],
        "parent_track_id": [pd.NA, "not-a-track"],
    })
    with pytest.raises(ValueError, match="parent_track_id contains non-numeric"):
        summarize_lineages(lineage)
