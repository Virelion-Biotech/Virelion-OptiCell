import numpy as np

from opticell.segmentation import ThresholdSegmenter, compare_backends, get_backend


def test_threshold_backend_matches_direct_engine():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[10:20, 10:20] = 255
    backend = ThresholdSegmenter(min_area=10, max_area_frac=0.2)
    result = backend.segment(image)
    assert result.count == 1
    assert result.method == "threshold"


def test_backend_factory_and_comparison():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[10:20, 10:20] = 255
    otsu = get_backend("otsu", min_area=10, max_area_frac=0.2)
    adaptive = get_backend("adaptive", min_area=10, max_area_frac=0.2)
    combined = compare_backends(image, {"otsu": otsu, "adaptive": adaptive})
    assert combined.labels.dtype == np.int32
    assert set(combined.member_counts) == {"otsu", "adaptive"}
