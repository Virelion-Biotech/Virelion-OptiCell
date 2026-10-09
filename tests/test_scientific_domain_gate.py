import pytest
import numpy as np
from opticell.domain_gate import require_domain_qualification
import qc_pipeline as qc


def test_missing_domain_and_qualification_fail_closed():
    for domain in [None, "unknown-modality"]:
        with pytest.raises(ValueError):
            require_domain_qualification(domain=domain, backend="threshold", report_path=None, report_sha256=None)


def test_cellpose_failure_never_silently_makes_a_threshold_mask(tmp_path, monkeypatch):
    import cv2

    path = tmp_path / "image.png"
    cv2.imwrite(str(path), np.zeros((20, 20), dtype=np.uint8))

    class Broken:
        def segment(self, *args, **kwargs):
            raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="fallback is disabled"):
        qc.analyze_image(str(path), cell_method="cellpose", cellpose_segmenter=Broken())
    result, mask = qc.analyze_image(
        str(path), cell_method="cellpose", cellpose_segmenter=Broken(), allow_fallback=True, return_segmentation=True
    )
    assert "SEGMENTATION_FALLBACK" in result.flags and mask.error


def test_report_is_bound_to_executing_fingerprint(tmp_path):
    import json
    import hashlib

    report = {
        "domain": "brightfield-hiPSC",
        "backend": "threshold",
        "model_id": "threshold-code",
        "model_sha256": "a" * 64,
        "dataset_id": "heldout",
        "dataset_version": "1",
        "protocol_sha256": "b" * 64,
        "gates": {"dice": True},
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    args = dict(
        domain=report["domain"],
        backend="threshold",
        report_path=path,
        report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert require_domain_qualification(**args, expected_model_sha256="a" * 64) == report
    with pytest.raises(ValueError, match="fingerprint"):
        require_domain_qualification(**args, expected_model_sha256="c" * 64)
