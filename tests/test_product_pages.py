from pathlib import Path


def test_product_pages_start_without_runtime_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTICELL_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("OPTICELL_DB", str(tmp_path / "workspace" / "workbench.sqlite3"))
    monkeypatch.setenv("OPTICELL_DATA_ROOT", str(tmp_path))

    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    pages = [
        root / "pages/1_Batch_Projects.py",
        root / "pages/2_Human_Review.py",
        root / "pages/3_Calibration_Validation.py",
        root / "pages/4_Large_Data_Operations.py",
    ]
    for page in pages:
        at = AppTest.from_file(page).run(timeout=60)
        assert not at.exception, (page.name, at.exception)
