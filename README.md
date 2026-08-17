# Virelion-OptiCell

**OptiCell 2.0** is a headless, research-oriented microscopy quality-control and quantitative cell-analysis engine from Virelion Biotech.

The project no longer depends on Streamlit or any GUI framework. It is built around a reusable Python API plus a reproducible command-line interface, so it can run locally, in notebooks, on servers, in CI, and inside larger bioinformatics pipelines.

## What changed in 2.0

- Removed Streamlit completely.
- Replaced the prototype QC-only flow with a reusable analysis engine.
- Added robust image hashing for provenance and duplicate detection.
- Added saturation and contrast metrics.
- Added segmentation-quality diagnostics rather than trusting cell counts blindly.
- Added adaptive dataset QC using median/MAD robust scores.
- Added a reusable Cellpose backend so the model is initialized once per batch instead of once per image.
- Added per-object morphology and intensity features.
- Added JSON export with analysis metadata.
- Added package metadata and the `opticell` CLI.
- Added automated tests and GitHub Actions CI.

## Core capabilities

### Acquisition QC

For every image, OptiCell can calculate:

| Metric | Purpose |
|---|---|
| Focus score | Variance of Laplacian; low values indicate blur |
| Mean brightness | Detects under/over-exposure |
| Brightness variability | Measures image contrast |
| Saturation fraction | Detects clipped pixels |
| Dimensions/channels/dtype | Captures acquisition structure |
| SHA-256 | Reproducibility and file identity |

Absolute QC flags include `BLURRY`, `TOO_DARK`, `TOO_BRIGHT`, `SATURATED`, `FEW_OR_NO_CELLS`, `TOO_MANY_CELLS`, `LOW_SEGMENTATION_QUALITY`, `SEGMENTATION_FALLBACK`, and `FAILED_TO_LOAD`.

### Segmentation

OptiCell currently exposes two interchangeable backends:

- `threshold`: fast classical segmentation using Otsu or local adaptive thresholding, morphology, and connected components.
- `cellpose`: optional deep-learning segmentation backend with a persistent model instance for batch processing.

Every backend returns a common `SegmentationResult`, making additional models easier to add later.

### Segmentation diagnostics

A cell count alone is not enough. OptiCell also measures:

- foreground fraction
- median object area
- coefficient of variation of object area
- border-object fraction
- tiny-object fraction
- merged-object fraction
- segmentation quality score

This helps identify images where a segmentation algorithm technically returned masks but the masks are not trustworthy.

### Quantitative cell features

`extract_object_features()` produces one row per object with:

- area
- perimeter
- circularity
- bounding box
- aspect ratio
- centroid
- mean intensity
- intensity standard deviation
- maximum intensity

This is the foundation for later phenotype, spatial, and multi-channel analyses.

### Adaptive dataset QC

Absolute thresholds are useful but microscope-dependent. OptiCell can therefore calculate robust dataset-relative scores using median/MAD statistics. Images that are strong outliers can be marked with `ADAPTIVE_OUTLIER` without deleting their original QC flags.

## Installation

### Standard installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e .
```

### With Cellpose

```bash
pip install -e ".[cellpose]"
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Command line

Analyze a directory recursively:

```bash
opticell /path/to/images -o qc_summary.csv
```

Generate CSV plus provenance JSON:

```bash
opticell /path/to/images \
  -o qc_summary.csv \
  --json qc_summary.json
```

Use Cellpose:

```bash
opticell /path/to/images \
  --cell-method cellpose \
  -o qc_summary.csv
```

Use local adaptive thresholding and dataset-level adaptive QC:

```bash
opticell /path/to/images \
  --adaptive-threshold \
  -o qc_summary.csv
```

Disable dataset-relative outlier scoring when strict absolute-threshold reproducibility is required:

```bash
opticell /path/to/images --no-adaptive-qc -o qc_summary.csv
```

The source checkout also supports:

```bash
python qc_pipeline.py /path/to/images -o qc_summary.csv
```

## Python API

```python
from qc_pipeline import (
    QCThresholds,
    analyze_folder,
    analyze_image,
    extract_object_features,
)

thresholds = QCThresholds(
    focus_min=100,
    brightness_min=25,
    brightness_max=230,
)

df = analyze_folder(
    "./experiment_images",
    thresholds=thresholds,
    cell_method="threshold",
    adaptive_qc=True,
)

result, segmentation = analyze_image(
    "./experiment_images/sample_01.tif",
    thresholds=thresholds,
    return_segmentation=True,
)

features = extract_object_features(
    # supply the same grayscale image used for segmentation
    gray_image,
    segmentation.labels,
)
```

## Output

The main CSV contains fields such as:

- `filename`, `path`, `sha256`
- `width`, `height`, `channels`, `dtype`, `ndim`
- `focus_score`
- `brightness_mean`, `brightness_std`, `contrast_std`
- `saturation_fraction`
- `estimated_cells`, `cell_method`
- `segmentation_quality`
- `median_cell_area`, `cell_area_cv`, `border_object_fraction`
- `adaptive_score`
- robust-z columns for adaptive QC
- `flags`
- `error`

JSON export additionally stores pipeline version, requested segmentation method, thresholds, input path, and the analysis records.

## Testing

Run the regression suite:

```bash
pytest
```

Compile-check the engine:

```bash
python -m compileall -q qc_pipeline.py
```

CI runs tests against Python 3.10, 3.11, and 3.12.

## Project structure

```text
Virelion-OptiCell/
├── qc_pipeline.py             # Core engine + CLI
├── make_test_images.py        # Synthetic microscopy fixtures
├── tests/
│   └── test_qc_pipeline.py    # Regression tests
├── .github/workflows/ci.yml   # Automated CI
├── pyproject.toml             # Packaging + CLI metadata
├── requirements.txt           # Runtime dependencies
└── LICENSE
```

## Important limitations

OptiCell is not yet a validated clinical or publication-grade measurement system. Classical segmentation can fail on crowded, low-contrast, highly variable, or unusually stained images. Cellpose is optional and should also be benchmarked on the specific cell type and imaging modality being analyzed.

Multi-dimensional TIFF input is currently reduced conservatively to a 2-D representation; full native C/Z/T analysis is a planned extension rather than something the current loader should pretend to support perfectly.

For scientific use, inspect segmentation quality and validate object-level measurements against an appropriate ground-truth subset before using results as experimental endpoints.
