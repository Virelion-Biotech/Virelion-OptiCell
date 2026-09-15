import numpy as np
import pytest

from tracking import link_frames


def test_tracking_rejects_negative_instance_labels():
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:4, 2:4] = -1
    with pytest.raises(ValueError, match="non-negative"):
        link_frames([labels])
