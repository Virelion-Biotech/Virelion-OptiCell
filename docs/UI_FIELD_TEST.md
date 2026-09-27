# UI field-test checklist

Automated AppTest coverage verifies application startup and a real image-upload/threshold segmentation path. A human field test is still useful for browser rendering, download behavior, and usability.

Run locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[ui,cellpose]'
streamlit run app_streamlit.py
```

Record whether these steps work without errors:

1. Upload a PNG or TIFF microscopy image.
2. Leave backend on `auto` and note the resolved backend shown by the UI.
3. Confirm the run-summary metrics, acceptance gate, acquisition QC, segmentation QC, object count, and overlay rendering.
4. Download the overlay PNG, per-object CSV, and run-summary JSON.
5. Confirm the run-summary JSON records the input SHA-256 plus requested/resolved backend and QC/acceptance outputs.
6. Repeat with `threshold` to verify the CPU path.
7. Upload a file with punctuation in its filename and confirm the filename is rendered as text rather than interpreted as HTML.

This checklist is usability feedback, not scientific validation.
