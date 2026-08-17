from pathlib import Path

import cv2
import numpy as np

from qc_pipeline import (
    QCThresholds,
    adaptive_dataset_qc,
    analyze_image,
    analyze_paths,
    compute_brightness,
    extract_object_features,
    find_images,
    robust_zscore,
    segment_threshold,
    to_grayscale_uint8,
)


def _write(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)
    assert ok


def test_grayscale_conversion_preserves_uint8_shape():
    image = np.zeros((32, 48), dtype=np.uint8)
    converted = to_grayscale_uint8(image)
    assert converted.dtype == np.uint8
    assert converted.shape == image.shape


def test_rgb_conversion_returns_grayscale():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    converted = to_grayscale_uint8(image)
    assert converted.shape == (20, 30)
    assert converted.dtype == np.uint8


def test_brightness_is_predictable():
    image = np.full((20, 20), 120, dtype=np.uint8)
    mean, std = compute_brightness(image)
    assert mean == 120.0
    assert std == 0.0


def test_threshold_segmentation_finds_separated_objects():
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(image, (25, 25), 8, 255, -1)
    cv2.circle(image, (75, 75), 9, 255, -1)
    result = segment_threshold(image, min_area=30, max_area_frac=0.1)
    assert result.count == 2
    assert result.labels.dtype == np.int32
    assert result.median_area > 0


def test_object_features_are_one_row_per_object():
    image = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(image, (25, 25), 7, 255, -1)
    cv2.circle(image, (55, 55), 7, 255, -1)
    result = segment_threshold(image, min_area=20, max_area_frac=0.2)
    features = extract_object_features(image, result.labels)
    assert len(features) == 2
    assert set(["area_px", "circularity", "centroid_x", "centroid_y"]).issubset(features.columns)


def test_analyze_image_flags_empty_scene(tmp_path):
    path = tmp_path / "empty.png"
    _write(path, np.zeros((64, 64), dtype=np.uint8))
    result = analyze_image(str(path))
    assert "TOO_DARK" in result.flags
    assert "FEW_OR_NO_CELLS" in result.flags
    assert result.error is None
    assert len(result.sha256) == 64


def test_find_images_is_case_insensitive_and_recursive(tmp_path):
    a = tmp_path / "a.PNG"
    bdir = tmp_path / "nested"
    bdir.mkdir()
    b = bdir / "b.tIfF"
    c = bdir / "ignore.txt"
    _write(a, np.zeros((10, 10), dtype=np.uint8))
    _write(b, np.zeros((10, 10), dtype=np.uint8))
    c.write_text("ignore")
    found = find_images(str(tmp_path))
    assert str(a) in found
    assert str(b) in found
    assert str(c) not in found
    assert len(found) == 2


def test_batch_analysis_keeps_duplicate_basenames(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    a = left / "same.png"
    b = right / "same.png"
    _write(a, np.zeros((20, 20), dtype=np.uint8))
    _write(b, np.full((20, 20), 100, dtype=np.uint8))
    df = analyze_paths([str(a), str(b)], adaptive_qc=False)
    assert len(df) == 2
    assert len(set(df["path"])) == 2


def test_adaptive_qc_adds_score_without_erasing_absolute_flags():
    df = analyze_paths([], adaptive_qc=False)
    # Use a minimal synthetic result table for direct adaptive-QC testing.
    import pandas as pd

    synthetic = pd.DataFrame(
        {
            "flags": ["TOO_DARK", "", ""],
            "focus_score": [10.0, 100.0, 100.0],
            "brightness_mean": [10.0, 100.0, 100.0],
            "saturation_fraction": [0.0, 0.0, 0.0],
            "contrast_std": [1.0, 10.0, 10.0],
            "estimated_cells": [2, 20, 21],
            "segmentation_quality": [50.0, 90.0, 91.0],
        }
    )
    result = adaptive_dataset_qc(synthetic)
    assert "adaptive_score" in result.columns
    assert "TOO_DARK" in result.loc[0, "flags"]


def test_robust_zscore_handles_zero_mad():
    series = __import__("pandas").Series([5.0, 5.0, 5.0])
    result = robust_zscore(series)
    assert np.all(result.to_numpy() == 0)


def test_invalid_thresholds_are_rejected():
    thresholds = QCThresholds(brightness_min=250, brightness_max=100)
    try:
        thresholds.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid thresholds should raise ValueError")
