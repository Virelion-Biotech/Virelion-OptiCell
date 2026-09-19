"""Automated Streamlit UI smoke tests.

These tests launch the real app with Streamlit's in-process AppTest harness.
"""

def test_streamlit_app_starts_without_runtime_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app_streamlit.py").run(timeout=60)

    assert not at.exception
    assert at.title
    assert at.title[0].value == "OptiCell — Easy UI"
    assert at.sidebar.selectbox
    assert at.sidebar.selectbox[0].value == "auto"
    assert at.file_uploader
