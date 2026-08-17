import numpy as np
import pandas as pd

from compartments import assign_nuclei_to_cells, compartment_features
from ensemble import threshold_ensemble, ensemble_from_results
from experiment import annotate_results, parse_metadata, plate_heatmap, summarize_groups
from phenotype import Rule, group_phenotype_summary, marker_positivity, score_cells
from tracking import TrackingConfig, link_frames, summarize_tracks


def test_compartment_assignment_and_ratios():
    cells = np.zeros((20, 20), dtype=np.int32); cells[2:18, 2:18] = 1
    nuclei = np.zeros_like(cells); nuclei[7:10, 7:10] = 1
    assigned = assign_nuclei_to_cells(cells, nuclei)
    assert assigned.iloc[0]["cell_label"] == 1
    features = compartment_features(np.full((20, 20), 10, dtype=np.uint8), cells, nuclei)
    assert features.iloc[0]["nucleus_area_px"] == 9
    assert 0 < features.iloc[0]["nucleus_to_cell_area_ratio"] < 1


def test_phenotype_rules_are_auditable():
    df = pd.DataFrame({"mean_intensity": [10, 100], "circularity": [0.2, 0.9]})
    scored = score_cells(df, [Rule("mean_intensity", 50, label="marker"), Rule("circularity", 0.8, label="round")])
    assert list(scored["phenotype_label"]) == ["negative", "positive"]
    assert "marker" in scored.loc[1, "phenotype_reasons"]
    assert group_phenotype_summary(scored)["positive_fraction"].iloc[0] == 0.5
    positive = marker_positivity(df, "mean_intensity", 50)
    assert int(positive["marker_positive"].sum()) == 1


def test_tracking_and_motion_summary():
    a = np.zeros((30, 30), dtype=np.int32); a[5:9, 5:9] = 1
    b = np.zeros_like(a); b[6:10, 7:11] = 1
    c = np.zeros_like(a); c[7:11, 9:13] = 1
    tracks = link_frames([a, b, c], TrackingConfig(max_distance_px=10))
    assert tracks["track_id"].nunique() == 1
    summary = summarize_tracks(tracks, pixel_size=0.5, frame_interval=2)
    assert len(summary) == 1
    assert summary.iloc[0]["path_length"] > 0


def test_experiment_metadata_and_plate_matrix():
    meta = parse_metadata("Plate2_MI_A07_t3.png")
    assert meta["plate"] == 2 and meta["well"] == "A07" and meta["timepoint"] == 3
    compact = parse_metadata("Plate2_MI_A07t3.png")
    assert compact["plate"] == 2 and compact["well"] == "A07" and compact["timepoint"] == 3
    df = pd.DataFrame({"filename": ["Plate2_A07_t0.png", "Plate2_B12_t0.png"], "cells": [10, 20], "condition": ["control", "MI"]})
    annotated = annotate_results(df)
    assert set(["plate", "well", "timepoint"]).issubset(annotated.columns)
    matrix = plate_heatmap(annotated, "cells")
    assert matrix.loc["A", 7] == 10
    summary = summarize_groups(annotated, ["condition"], ["cells"])
    assert set(summary["condition"]) == {"control", "MI"}


def test_ensemble_disagreement_is_exposed():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[15:25, 15:25] = 255; image[40:50, 40:50] = 255
    result = threshold_ensemble(image, min_area=20, max_area_frac=0.2)
    assert set(result.member_counts) == {"otsu", "adaptive"}
    assert result.labels.dtype == np.int32
    assert 0 <= result.agreement_fraction <= 1


def test_ensemble_empty_validation():
    import pytest
    with pytest.raises(ValueError):
        ensemble_from_results([])
