import numpy as np
import pandas as pd

from image_io import ImageStack, canonicalize_axes, project_z, select_channel, select_time
from quantitative import (
    add_spatial_features,
    apply_background_correction,
    channel_summary,
    colocated_fraction,
    normalized_colocalization,
    object_channel_intensity,
    summarize_spatial_features,
)


def test_spatial_features_and_summary():
    features = pd.DataFrame(
        {
            "centroid_x": [10.0, 20.0, 80.0],
            "centroid_y": [10.0, 20.0, 80.0],
        }
    )
    enriched = add_spatial_features(features, (100, 100))
    assert "nearest_neighbor_distance_px" in enriched.columns
    assert enriched["nearest_neighbor_distance_px"].iloc[0] > 0
    summary = summarize_spatial_features(features, (100, 100))
    assert summary["object_count"] == 3.0
    assert summary["density_per_100k_px"] == 300.0


def test_channel_summary_and_object_intensity():
    image = np.zeros((10, 10, 2), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 100
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:5, 2:5] = 1
    summary = channel_summary(image)
    assert len(summary) == 2
    measured = object_channel_intensity(image, labels)
    assert len(measured) == 2
    assert measured.loc[measured["channel"] == 1, "mean_intensity"].iloc[0] == 100.0


def test_colocalization_helpers():
    a = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    b = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    assert colocated_fraction(a, b) == 0.5
    assert normalized_colocalization(a, a) > 0.99


def test_background_correction_is_bounded():
    image = np.full((20, 20), 100, dtype=np.uint8)
    image[8:12, 8:12] = 180
    corrected = apply_background_correction(image, radius=2)
    assert corrected.dtype == np.uint8
    assert corrected.min() >= 0
    assert corrected.max() <= 255


def test_dimension_aware_stack_helpers():
    data = np.arange(2 * 3 * 2 * 4 * 5).reshape(2, 3, 2, 4, 5)
    stack = ImageStack(data=data, axes="TZCYX", path="x.tif", dtype=str(data.dtype), shape=data.shape)
    channel = select_channel(stack, 1)
    assert channel.shape == (2, 3, 4, 5)
    t0 = select_time(stack, 1)
    assert t0.axes == "ZCYX"
    assert t0.data.shape == (3, 2, 4, 5)
    projected = project_z(stack, "max")
    assert projected.axes == "TCYX"
    assert projected.data.shape == (2, 2, 4, 5)


def test_canonicalize_axes():
    data = np.zeros((3, 2, 4, 5), dtype=np.uint16)
    stack = ImageStack(data=data, axes="CZYX", path="x.tif", dtype="uint16", shape=data.shape)
    canonical = canonicalize_axes(stack)
    assert canonical.axes == "ZCYX"
    assert canonical.data.shape == (2, 3, 4, 5)
