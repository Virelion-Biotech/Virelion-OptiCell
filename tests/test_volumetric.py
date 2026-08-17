import numpy as np

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
