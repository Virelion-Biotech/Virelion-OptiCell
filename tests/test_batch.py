from pathlib import Path

import cv2
import numpy as np
import pytest

from opticell.batch import BatchConfig, analyze_paths_parallel


def _write(path: Path, value: int) -> None:
    image = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(image, (24, 24), 7, value, -1)
    assert cv2.imwrite(str(path), image)


def test_parallel_batch_preserves_sorted_paths_and_progress(tmp_path):
    paths = []
    for name, value in [("b.png", 180), ("a.png", 120), ("c.png", 220)]:
        path = tmp_path / name
        _write(path, value)
        paths.append(str(path))

    progress = []
    frame = analyze_paths_parallel(
        paths,
        config=BatchConfig(workers=2, adaptive_qc=False),
        progress_callback=lambda done, total, name: progress.append((done, total, name)),
    )
    assert list(frame["path"]) == sorted(str(p.resolve()) for p in [Path(p) for p in paths])
    assert len(frame) == 3
    assert len(progress) == 3
    assert sorted(done for done, _, _ in progress) == [1, 2, 3]


def test_parallel_batch_validates_configuration():
    with pytest.raises(ValueError):
        BatchConfig(workers=0).validate()
    with pytest.raises(ValueError):
        BatchConfig(cell_method="unknown").validate()
