import numpy as np
import pytest

from preprocessing import detect_hot_pixels, flat_field_correct, subtract_background


def test_flat_field_correction_preserves_integer_dtype_and_clips():
    image = np.full((3, 3), 60000, dtype=np.uint16)
    reference = np.full((3, 3), 30000, dtype=np.uint16)
    reference[1, 1] = 1
    corrected = flat_field_correct(image, reference)
    assert corrected.dtype == np.uint16
    assert int(corrected.max()) == np.iinfo(np.uint16).max


def test_flat_field_rejects_nonfinite_reference():
    image = np.ones((3, 3), dtype=np.float32)
    reference = np.ones((3, 3), dtype=np.float32)
    reference[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        flat_field_correct(image, reference)


def test_subtract_background_does_not_wrap_integer_output():
    image = np.full((20, 20), 65535, dtype=np.uint16)
    corrected = subtract_background(image, sigma=1.0)
    assert corrected.dtype == np.uint16
    assert np.all(corrected == 0)


def test_detect_hot_pixels_is_local_and_returns_boolean_mask():
    image = np.full((9, 9), 100, dtype=np.uint8)
    image[4, 4] = 255
    mask = detect_hot_pixels(image)
    assert mask.dtype == bool
    assert bool(mask[4, 4])
