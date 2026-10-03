import numpy as np
import pytest

from volumetric import nearest_neighbor_distances_3d, summarize_volume, volume_features


def test_volume_features_and_summary():
    labels = np.zeros((6, 8, 10), dtype=np.int32)
    labels[1:3, 1:4, 2:5] = 1
    labels[3:5, 5:7, 6:9] = 2
    features = volume_features(labels, voxel_size=(2.0, 1.0, 0.5))
    assert len(features) == 2
    assert features[0]["volume"] == 18.0
    summary = summarize_volume(labels, voxel_size=(2.0, 1.0, 0.5))
    assert summary["object_count"] == 2.0
    assert 0 < summary["volume_fraction"] < 1


def test_3d_nearest_neighbor_distances():
    features = [
        {"centroid_z_um": 0.0, "centroid_y_um": 0.0, "centroid_x_um": 0.0},
        {"centroid_z_um": 0.0, "centroid_y_um": 0.0, "centroid_x_um": 3.0},
        {"centroid_z_um": 0.0, "centroid_y_um": 4.0, "centroid_x_um": 0.0},
    ]
    distances = nearest_neighbor_distances_3d(features)
    assert np.allclose(distances, [3.0, 3.0, 4.0])


def test_single_voxel_volume_and_surface_area_in_um_units():
    labels = np.zeros((3, 3, 3), dtype=np.int32)
    labels[1, 1, 1] = 1
    rows = volume_features(labels, voxel_size=(2.0, 3.0, 4.0))
    assert rows[0]["volume"] == 24.0
    assert rows[0]["surface_area_approx"] == 2 * (3 * 4 + 2 * 4 + 2 * 3)
    assert rows[0]["centroid_x_um"] == 4.0
    assert rows[0]["centroid_y_um"] == 3.0
    assert rows[0]["centroid_z_um"] == 2.0


def test_volume_summary_density_is_per_mm3():
    labels = np.zeros((10, 10, 10), dtype=np.int32)
    labels[1, 1, 1] = 1
    summary = summarize_volume(labels, voxel_size=(100.0, 100.0, 100.0))
    assert np.isclose(summary["volume_fraction"], 0.001)
    assert np.isclose(summary["object_density_per_mm3"], 1.0)


def test_volumetric_rejects_invalid_spacing_and_negative_labels():
    labels = np.zeros((2, 2, 2), dtype=np.int32)
    labels[0, 0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        volume_features(labels)
    with pytest.raises(ValueError, match="finite positive"):
        volume_features(np.zeros((2, 2, 2), dtype=np.int32), (1.0, np.inf, 1.0))


def test_3d_nearest_neighbor_requires_finite_physical_coordinates():
    with pytest.raises(ValueError, match="finite"):
        nearest_neighbor_distances_3d([
            {"centroid_z_um": 0.0, "centroid_y_um": 0.0, "centroid_x_um": np.nan},
            {"centroid_z_um": 1.0, "centroid_y_um": 0.0, "centroid_x_um": 0.0},
        ])
