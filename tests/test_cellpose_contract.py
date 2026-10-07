import numpy as np
import pytest

import app_streamlit
from qc_pipeline import SegmentationResult


def test_missing_checkpoint_cannot_be_substituted_with_default(monkeypatch):
    from types import SimpleNamespace
    import qc_pipeline

    calls = []
    def constructor(**kwargs):
        calls.append(kwargs)
        return object()
    monkeypatch.setattr(qc_pipeline, "_HAS_CELLPOSE", True)
    monkeypatch.setattr(qc_pipeline, "_cellpose_models", SimpleNamespace(
        MODEL_NAMES=["cpsam"], get_user_models=lambda: [], CellposeModel=constructor))
    with pytest.raises(ValueError, match="Unknown or missing Cellpose checkpoint"):
        qc_pipeline.CellposeSegmenter("missing-checkpoint.pt").model
    assert calls == []


def test_checkpoint_load_failure_cannot_retry_with_default(monkeypatch):
    from types import SimpleNamespace
    import qc_pipeline

    calls = []
    def constructor(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("Corrupt requested weights")
    monkeypatch.setattr(qc_pipeline, "_HAS_CELLPOSE", True)
    monkeypatch.setattr(qc_pipeline, "_cellpose_models", SimpleNamespace(
        MODEL_NAMES=["cpsam"], get_user_models=lambda: [], CellposeModel=constructor))
    with pytest.raises(RuntimeError, match="Corrupt requested weights"):
        qc_pipeline.CellposeSegmenter("cpsam").model
    assert calls == [{"pretrained_model": "cpsam"}]


def test_gpu_request_cannot_be_reported_as_gpu_after_cpu_resolution(monkeypatch):
    from types import SimpleNamespace
    import qc_pipeline

    monkeypatch.setattr(qc_pipeline, "_HAS_CELLPOSE", True)
    monkeypatch.setattr(qc_pipeline, "_cellpose_models", SimpleNamespace(
        MODEL_NAMES=["cpsam"], get_user_models=lambda: [],
        CellposeModel=lambda **kwargs: SimpleNamespace(gpu=False)))
    segmenter = qc_pipeline.CellposeSegmenter("cpsam", gpu=True)
    with pytest.raises(RuntimeError, match="resolved to CPU"):
        segmenter.model
    assert segmenter._model is None


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
