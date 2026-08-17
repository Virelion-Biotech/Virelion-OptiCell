import numpy as np

from validation import (
    benchmark_segmentation,
    binary_dice,
    binary_iou,
    count_error,
    instance_metrics,
    match_instance_centroids,
    segmentation_pixel_metrics,
)


def test_binary_metrics_perfect_and_partial():
    truth = np.zeros((10, 10), dtype=np.uint8)
    truth[2:6, 2:6] = 1
    pred = truth.copy()
    assert binary_iou(pred, truth) == 1.0
    assert binary_dice(pred, truth) == 1.0
    partial = np.zeros_like(truth)
    partial[2:5, 2:6] = 1
    metrics = segmentation_pixel_metrics(partial, truth)
    assert 0 < metrics["iou"] < 1
    assert 0 < metrics["recall"] <= 1


def test_instance_matching_and_count_error():
    truth = np.zeros((50, 50), dtype=np.int32)
    truth[5:10, 5:10] = 1
    truth[30:35, 30:35] = 2
    pred = np.zeros_like(truth)
    pred[6:11, 6:11] = 1
    pred[30:35, 30:35] = 2
    pred[40:45, 40:45] = 3
    tp, fp, fn = match_instance_centroids(pred, truth, max_distance_px=5)
    assert (tp, fp, fn) == (2, 1, 0)
    metrics = instance_metrics(pred, truth, max_distance_px=5)
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["f1"] < 1
    assert count_error(3, 2)["absolute_count_error"] == 1


def test_benchmark_aggregation():
    truth = np.zeros((20, 20), dtype=np.int32)
    truth[4:8, 4:8] = 1
    pred = truth.copy()
    result = benchmark_segmentation([pred], [truth])
    assert result["n_images"] == 1
    assert result["pixel_dice_mean"] == 1.0
    assert result["instance_f1_mean"] == 1.0
