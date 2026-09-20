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
3. Confirm acquisition QC, segmentation QC, object count, and overlay rendering.
4. Download the overlay PNG and per-object CSV.
5. Repeat with `threshold` to verify the CPU path.

This checklist is usability feedback, not scientific validation.
