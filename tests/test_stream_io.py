import types

import numpy as np
import pytest
import tifffile

from stream_io import iter_tiff_frames
import stream_io


def test_iter_tiff_frames_preserves_logical_leading_axis(tmp_path):
    path = tmp_path / "tz.tif"
    data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    tifffile.imwrite(path, data, metadata={"axes": "TZYX"})

    frames = list(iter_tiff_frames(str(path)))

    assert len(frames) == 2
    assert [frame.shape for frame in frames] == [(3, 4, 5)] * 2
    assert all(frame.dtype == data.dtype for frame in frames)
    assert np.array_equal(frames[0], data[0])
    assert np.array_equal(frames[1], data[1])


def test_iter_tiff_frames_rejects_ambiguous_page_layout(monkeypatch):
    class FakePage:
        def asarray(self):
            return np.zeros((3, 4), dtype=np.uint8)

    class FakeSeries:
        shape = (2, 3, 4)
        dtype = np.dtype(np.uint8)
        pages = [FakePage()]

    class FakeTiff:
        series = [FakeSeries()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    fake_tifffile = types.SimpleNamespace(TiffFile=lambda _path: FakeTiff())
    monkeypatch.setattr(stream_io, "tifffile", fake_tifffile)

    with pytest.raises(ValueError, match="page layout is ambiguous"):
        list(iter_tiff_frames("fake.tif"))
