# Virelion-OptiCell

A small application by Virelion Biotech for batch quality-control of microscopy image datasets.
Drop in a folder of images and get automatic checks for **focus (blur)**,
**brightness**, and **estimated cell counts**, plus dataset-wide histograms
and a downloadable summary CSV.

## What it does

For every image in a folder (`.png`, `.jpg`/`.jpeg`, `.tif`/`.tiff`, `.bmp`):

| Check | Method |
|---|---|
| **Focus / sharpness** | Variance of the Laplacian — low variance = blurry |
| **Brightness** | Mean pixel intensity, rescaled to 0–255 for any bit depth |
| **Estimated cell count** | Otsu auto-thresholding → morphological cleanup → connected-component labeling. Optional [Cellpose](https://www.cellpose.org/) backend if installed, for a real deep-learning segmentation count |
| **File / image stats** | Width, height, channels, dtype, file size |

Images are flagged automatically (`BLURRY`, `TOO_DARK`, `TOO_BRIGHT`,
`FEW_OR_NO_CELLS`, `FAILED_TO_LOAD`) using thresholds you can tune in the
sidebar, so you can scan hundreds of images and jump straight to the
suspicious ones instead of opening every file by hand.

## Project layout

```
microscopy_qc/
├── qc_pipeline.py     # Core analysis engine (no GUI deps — usable standalone/CLI)
├── app.py             # Streamlit drag-and-drop GUI
├── make_test_images.py# Generates a few synthetic test images (sharp/blurry/dark/bright/empty)
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the GUI

```bash
streamlit run app.py
```

This opens a browser tab where you can:
1. **Drag & drop** a batch of image files (or point at a **local folder path**
   for large datasets — the folder-path option only works when running
   Streamlit on your own machine, since a browser can't see local paths).
2. Adjust QC thresholds in the sidebar.
3. Click **Run QC analysis**.
4. Browse the results table (flagged rows highlighted), dataset-wide
   histograms, and a per-image preview with detected-cell overlay.
5. Download the summary CSV.

## Run from the command line (no GUI)

```bash
python3 qc_pipeline.py /path/to/images -o qc_summary.csv
```

Optional flags: `--focus-min`, `--brightness-min`, `--brightness-max`,
`--cell-method {threshold,cellpose}`.

## Try it with synthetic test data

```bash
python3 make_test_images.py          # writes test_images/ (sharp, blurry, dark, bright, empty, 16-bit)
python3 qc_pipeline.py test_images -o qc_summary_test.csv
```

## Output CSV columns

| Column | Description |
|---|---|
| `filename`, `path` | File identity |
| `width`, `height`, `channels`, `dtype` | Image dimensions/format |
| `file_size_kb` | File size on disk |
| `focus_score` | Variance of Laplacian (higher = sharper) |
| `brightness_mean`, `brightness_std` | Intensity stats, 0–255 scale |
| `estimated_cells` | Connected-component (or Cellpose) count |
| `cell_method` | Which method produced the count |
| `flags` | Semicolon-separated QC flags, empty if none |
| `error` | Populated only if the image failed to load |

## Notes & limitations

- The default cell counter is **classical image processing**, not a trained
  segmentation model — it works well for reasonably distinct, non-overlapping
  cells/nuclei on a roughly uniform background, but will under-count dense,
  touching clusters and can be thrown off by very unusual staining. For
  publication-grade counts, install `cellpose` (see `requirements.txt`) and
  switch the method in the sidebar.
- Multi-page/Z-stack TIFFs are collapsed to a single image via max-intensity
  projection before analysis.
- Flag thresholds (`focus_min`, `brightness_min/max`, etc.) are dataset- and
  microscope-dependent starting points — run once on a known-good batch of
  images and adjust the sidebar values to match your normal range.
