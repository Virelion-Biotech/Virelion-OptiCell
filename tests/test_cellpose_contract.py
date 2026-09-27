import numpy as np

import app_streamlit
from qc_pipeline import SegmentationResult


class _FakeCellpose:
    def segment(self, gray, **kwargs):
        labels = np.zeros_like(gray, dtype=np.int32)
        labels[10:30, 10:30] = 1
        return SegmentationResult(
            count=1,
            labels=labels,
            method="cellpose:fake",
            foreground_fraction=float((labels > 0).mean()),
            median_area=400.0,
            area_cv=0.0,
            border_fraction=0.0,
            tiny_object_fraction=0.0,
            merged_object_fraction=0.0,
            quality_score=95.0,
            error=None,
        )


def test_cellpose_pipeline_contract_with_injected_segmenter(monkeypatch):
    image = np.zeros((64, 64), dtype=np.uint8)
    image[10:30, 10:30] = 180
    result = app_streamlit.run_pipeline(image, "cellpose", cellpose_segmenter=_FakeCellpose())
    assert result["backend_resolved"] == "cellpose"
    assert result["seg"].count == 1
    assert result["accept"] is not None


def test_auto_cellpose_contract_when_available(monkeypatch):
    monkeypatch.setattr(app_streamlit, "_HAS_CELLPOSE", True)
    image = np.zeros((64, 64), dtype=np.uint8)
    image[10:30, 10:30] = 180
    result = app_streamlit.run_pipeline(image, "auto", cellpose_segmenter=_FakeCellpose())
    assert result["backend_resolved"] == "cellpose"
    assert result["seg"].method == "cellpose:fake"
