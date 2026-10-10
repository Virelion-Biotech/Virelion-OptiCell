"""Regression cases for nuclei larger than the legacy local threshold window."""
import cv2
import numpy as np
import pytest

from opticell.fluorescence import segment_fluorescence
from opticell.segmentation import get_backend


def test_heterogeneous_bright_nuclei_keep_dim_objects_and_fill_interiors():
    image = np.zeros((240, 320), np.uint8)
    truth = np.zeros(image.shape, np.int32)
    for label, (x, intensity) in enumerate([(70, 180), (235, 25)], 1):
        cv2.circle(image, (x, 120), 35, intensity, -1)
        cv2.circle(image, (x, 120), 12, 0, -1)
        cv2.circle(truth, (x, 120), 35, label, -1)
    result = segment_fluorescence(image)
    assert result.count == 2
    assert result.labels[120, 70] > 0 and result.labels[120, 235] > 0
    intersection = np.logical_and(result.labels > 0, truth > 0).sum()
    union = np.logical_or(result.labels > 0, truth > 0).sum()
    assert intersection / union > 0.9
    assert result.method == "fluorescence:log_otsu_watershed"


def test_touching_nuclei_are_split():
    image = np.zeros((180, 220), np.uint8)
    cv2.circle(image, (80, 90), 30, 100, -1)
    cv2.circle(image, (130, 90), 30, 100, -1)
    result = segment_fluorescence(image)
    assert result.count == 2
    assert result.labels[90, 80] != result.labels[90, 130]


@pytest.mark.parametrize("intensity", [0, 80, 255])
def test_uniform_image_has_no_objects(intensity):
    result = segment_fluorescence(np.full((100, 100), intensity, np.uint8))
    assert result.count == 0
    assert not result.labels.any()


def test_fluorescence_api_and_area_limits():
    image = np.zeros((100, 100), np.uint8)
    cv2.circle(image, (50, 50), 20, 100, -1)
    backend = get_backend("fluorescence", max_area_frac=0.01)
    assert backend.segment(image).count == 0
    with pytest.raises(ValueError, match="uint8"):
        backend.segment(image.astype(np.uint16))
    with pytest.raises(ValueError, match="area limits"):
        backend.segment(image, min_area=0)
