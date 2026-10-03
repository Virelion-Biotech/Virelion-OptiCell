"""Automated Streamlit UI smoke tests."""
from pathlib import Path

import cv2
import numpy as np


def _synthetic_png_bytes() -> bytes:
    image = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(image, (35, 45), 10, 220, -1)
    cv2.circle(image, (88, 82), 12, 220, -1)
    cv2.rectangle(image, (55, 20), (72, 38), 220, -1)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("failed to encode synthetic PNG")
    return encoded.tobytes()


def test_streamlit_app_starts_without_runtime_errors():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app_streamlit.py"
    at = AppTest.from_file(app_path).run(timeout=60)

    assert not at.exception
    assert any("OptiCell Research Workbench" in getattr(md, "value", "") for md in at.markdown)
    assert at.sidebar.selectbox
    assert at.sidebar.selectbox[0].value == "auto"
    assert at.file_uploader


def test_streamlit_threshold_workflow_accepts_uploaded_image():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app_streamlit.py"
    at = AppTest.from_file(app_path).run(timeout=60)

    at.sidebar.selectbox[0].select("threshold")
    at.file_uploader[0].upload("synthetic.png", _synthetic_png_bytes(), "image/png")
    at.run(timeout=60)

    assert not at.exception
    assert not at.error
    assert any(getattr(metric, "label", "") == "Objects" for metric in at.metric)
    assert any(button.label == "Download overlay PNG" for button in at.download_button)
    assert any(button.label == "Download run summary JSON" for button in at.download_button)


def test_confidence_flags_are_formatted_as_tokens_not_characters():
    from app_streamlit import _format_confidence_flags

    assert _format_confidence_flags("LOW_FOCUS;SPARSE_FG;ZERO_OBJECTS") == (
        "LOW_FOCUS, SPARSE_FG, ZERO_OBJECTS"
    )
    assert _format_confidence_flags("") == ""


def test_streamlit_escapes_uploaded_filename_in_custom_html():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app_streamlit.py"
    at = AppTest.from_file(app_path).run(timeout=60)

    at.sidebar.selectbox[0].select("threshold")
    at.file_uploader[0].upload("<b>unsafe</b>.png", _synthetic_png_bytes(), "image/png")
    at.run(timeout=60)

    assert not at.exception
    html_values = [getattr(md, "value", "") for md in at.markdown]
    assert any("&lt;b&gt;unsafe&lt;/b&gt;.png" in value for value in html_values)
    assert not any("<span class=\"pill\">File · <b>unsafe</b>.png" in value for value in html_values)
