import numpy as np

from validation import benchmark_segmentation, instance_metrics, segmentation_pixel_metrics


def test_3d_pixel_and_instance_metrics():
    truth = np.zeros((12, 16, 16), dtype=np.int32)
    pred = np.zeros_like(truth)
    truth[2:5, 2:5, 2:5] = 1
    truth[7:10, 9:12, 9:12] = 2
    pred[2:5, 2:5, 2:5] = 1
    pred[7:10, 9:12, 9:12] = 2
    metrics = segmentation_pixel_metrics(pred > 0, truth > 0)
    assert metrics["iou"] == 1.0
    assert instance_metrics(pred, truth, max_distance_px=2)["f1"] == 1.0
    benchmark = benchmark_segmentation([pred], [truth], max_distance_px=2)
    assert benchmark["n_images"] == 1
    assert benchmark["instance_f1_mean"] == 1.0
