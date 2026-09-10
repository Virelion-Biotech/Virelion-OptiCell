import numpy as np
import pytest

from volumetric_segmentation import segment_threshold_3d


def test_segment_threshold_3d_finds_two_objects():
    volume = np.zeros((24, 32, 32), dtype=np.uint8)
    volume[3:8, 4:9, 4:9] = 255
    volume[14:20, 20:26, 20:26] = 255
    result = segment_threshold_3d(volume, min_volume_voxels=20)
    assert result.count == 2
    assert result.labels.dtype == np.int32
    assert result.foreground_fraction > 0


def test_segment_threshold_3d_supports_explicit_normalization_modes():
    volume = np.zeros((12, 16, 16), dtype=np.float32)
    volume[2:5, 2:5, 2:5] = 1.0
    percentile = segment_threshold_3d(volume, min_volume_voxels=5, normalization="percentile")
    none = segment_threshold_3d(volume, min_volume_voxels=5, normalization="none")
    assert percentile.count == 1
    assert none.count == 1


def test_segment_threshold_3d_rejects_invalid_normalization():
    with pytest.raises(ValueError, match="normalization"):
        segment_threshold_3d(np.zeros((4, 4, 4), dtype=np.float32), normalization="unknown")
    with pytest.raises(ValueError, match="values in \[0, 1\]"):
        segment_threshold_3d(np.ones((4, 4, 4), dtype=np.float32) * 10, normalization="none")


def test_segment_threshold_3d_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        segment_threshold_3d(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        segment_threshold_3d(np.zeros((4, 4, 4), dtype=np.uint8), voxel_size=(1, 1, 0))


def test_segment_threshold_3d_constant_volume_is_empty():
    result = segment_threshold_3d(np.full((8, 8, 8), 50, dtype=np.uint8), min_volume_voxels=2)
    assert result.count == 0
    assert np.count_nonzero(result.labels) == 0
