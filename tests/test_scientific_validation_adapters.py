"""Counterexamples for measured-data admission and reference scoring."""

import cv2
import numpy as np
import pytest
from pycocotools import mask as coco_masks
from scripts.run_bbbc039_validation import decode_bbbc039_mask
from scripts.run_bbbc038_validation import decode_bbbc038_instance_masks
from scripts.run_livecell_validation import _polygon_to_mask, _rle_to_mask
from validation import instance_iou_metrics, instance_metrics


def test_touching_differently_colored_nuclei_remain_separate(tmp_path):
    rgb = np.zeros((8, 10, 3), np.uint8)
    rgb[2:6, 2:5, 0] = 1
    rgb[2:6, 5:8, 0] = 2
    p = tmp_path / "mask.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    labels = decode_bbbc039_mask(p)
    assert labels.max() == 2
    assert np.array_equal(labels > 0, rgb[:, :, 0] > 0)
    assert labels[3, 4] != labels[3, 5]


def test_distinct_disconnected_objects_with_reused_color(tmp_path):
    img = np.zeros((12, 12, 3), np.uint8)
    img[1:3, 1:3, 2] = 1
    img[8:10, 8:10, 2] = 1
    p = tmp_path / "reused.png"
    cv2.imwrite(str(p), img)
    assert decode_bbbc039_mask(p).max() == 2


def test_coco_fractional_contours_match_official_decoder():
    polygon = [1.2, 1.3, 5.7, 1.6, 4.6, 6.3, 1.4, 5.8]
    ref = coco_masks.decode(coco_masks.merge(coco_masks.frPyObjects([polygon], 9, 9)))
    assert np.array_equal(_polygon_to_mask(polygon, 9, 9), ref)


def test_bad_rle_and_missing_reference_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="dimensions/counts"):
        _rle_to_mask({"size": [4, 4], "counts": [3, 4]}, 4, 4)
    with pytest.raises(FileNotFoundError, match="reference"):
        decode_bbbc038_instance_masks(tmp_path, (4, 4))


def test_centroid_match_does_not_establish_shape_accuracy():
    truth = np.zeros((21, 21), np.int32)
    truth[7:14, 7:14] = 99
    prediction = np.ones_like(truth) * 8000000
    assert instance_metrics(prediction, truth)["f1"] == 1.0
    assert instance_iou_metrics(prediction, truth)["instance_iou_f1"] == 0.0


def test_iou_empty_and_sparse_label_semantics():
    empty = np.zeros((4, 4), np.int32)
    one = empty.copy()
    one[1, 1] = 1000000000
    assert instance_iou_metrics(empty, empty)["instance_iou_f1"] == 1.0
    assert instance_iou_metrics(one, empty)["instance_iou_false_positives"] == 1.0
    assert instance_iou_metrics(empty, one)["instance_iou_false_negatives"] == 1.0
    assert instance_iou_metrics(one, one)["instance_iou_f1"] == 1.0


def test_bbbc006_random_image_uuids_do_not_split_physical_site():
    from pathlib import Path
    from scripts.run_bbbc006_focus_qc import site_key, spearman_corr

    a = Path("z_00/mcf-z-stacks-03212011_a02_s1_w1uuid-one.tif")
    b = Path("z_16/mcf-z-stacks-03212011_a02_s1_w1uuid-two.tif")
    assert site_key(a) == site_key(b)
    assert spearman_corr(np.array([1.0, 1.0, 2.0]), np.array([2.0, 2.0, 3.0])) == 1.0
    assert spearman_corr(np.ones(3), np.array([1.0, 2.0, 3.0])) is None


def test_vectorized_centroids_preserve_sparse_2d_and_3d_results():
    from validation import _centroids_from_labels

    for shape in [(9, 11), (3, 9, 11)]:
        rng = np.random.default_rng(921)
        labels = rng.choice([0, 7, 1000000000], size=shape)
        expected = np.array([np.argwhere(labels == k).mean(0) for k in [7, 1000000000]])
        assert np.array_equal(_centroids_from_labels(labels), expected)


def test_vectorized_diagnostics_preserve_area_and_border_statistics():
    from qc_pipeline import _segmentation_diagnostics

    labels = np.zeros((12, 14), np.int32)
    labels[0:3, 1:4] = 7000000
    labels[4:7, 5:8] = 99
    result = _segmentation_diagnostics(labels, labels.shape, 1, 0.25)
    assert result[0] == 18 / 168
    assert result[1] == 9
    assert result[2] == 0
    assert result[3] == 0.5
    assert result[4:6] == (0.0, 0.0)
    assert result[6] == 85.0


def test_ctc_seg_scores_only_annotated_objects_and_requires_majority_overlap():
    from scripts.score_ctc_seg import seg_object_scores

    truth = np.zeros((8, 8), np.int32)
    truth[2:4, 2:4] = 1
    prediction = truth.copy()
    prediction[6:8, 6:8] = 2  # Unannotated cells are not false positives in CTC SEG.
    assert seg_object_scores(prediction, truth) == [1.0]
    prediction[2, 2:4] = 0  # Exactly 50 percent GT coverage does not qualify.
    assert seg_object_scores(prediction, truth) == [0.0]


def test_adaptive_bright_objects_do_not_become_background_halos():
    from qc_pipeline import segment_threshold
    from validation import paired_segmentation_metrics

    image = np.zeros((80, 100), np.uint8)
    truth = np.zeros_like(image, dtype=np.int32)
    for index, (x, y) in enumerate([(28, 30), (70, 50)], 1):
        cv2.circle(image, (x, y), 8, 200, -1)
        cv2.circle(truth, (x, y), 8, index, -1)
    result = segment_threshold(image, adaptive=True)
    metrics = paired_segmentation_metrics(result.labels, truth)
    assert result.count == 2
    assert metrics["pixel_dice"] > 0.9
    assert metrics["instance_iou_f1"] == 1.0


def test_missing_bbbc039_reference_is_not_silently_excluded(tmp_path):
    from scripts.run_bbbc039_validation import find_pairs

    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    cv2.imwrite(str(images / "unpaired.tif"), np.zeros((8, 8), np.uint8))
    with pytest.raises(FileNotFoundError, match="missing reference"):
        find_pairs(images, masks)


def test_ambiguous_source_files_fail_without_conflating_repeated_coco_ids(tmp_path):
    from scripts.run_livecell_validation import build_image_index

    for folder in ["first", "second"]:
        path = tmp_path / folder
        path.mkdir()
        cv2.imwrite(str(path / "same.tif"), np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError, match="Ambiguous source image"):
        build_image_index(tmp_path)


def test_undefined_empty_reference_ratio_serializes_as_null_without_changing_absolute_error():
    import json
    from validation import count_error, json_ready_metrics, _finite_mean

    error = count_error(3, 0)
    assert np.isnan(error["relative_count_error"])
    encoded = json.dumps(json_ready_metrics(dict(per_image=[error])), allow_nan=False)
    decoded = json.loads(encoded)["per_image"][0]
    assert decoded["relative_count_error"] is None
    assert decoded["absolute_count_error"] == 3.0
    assert _finite_mean([None, 0.5]) == 0.5
