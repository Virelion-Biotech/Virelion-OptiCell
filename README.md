# Virelion-OptiCell

**OptiCell 2.1** is a headless, research-oriented microscopy quality-control and quantitative cell-analysis toolkit from Virelion Biotech.

There is **no Streamlit and no GUI dependency**. OptiCell is built as a reusable Python API plus CLI for local analysis, notebooks, servers, CI, and larger bioinformatics pipelines.

## What changed

### 2.0 — analysis engine
- Removed Streamlit completely.
- Added robust image hashing and provenance fields.
- Added focus, brightness, contrast, and saturation QC.
- Added segmentation diagnostics instead of trusting cell counts blindly.
- Added adaptive dataset QC using robust median/MAD scores.
- Added reusable Cellpose segmentation with batch-level model reuse.
- Added per-object morphology and intensity measurements.
- Added deterministic CSV and JSON export.
- Added package metadata, CLI, tests, and GitHub Actions CI.

### 2.1 — quantitative microscopy layer
- Added **native dimension-aware TIFF I/O** with explicit C/Z/T axis handling.
- Added max/mean/median Z projections without silently destroying other axes.
- Added explicit timepoint and channel selection.
- Added **spatial phenotype features**: nearest-neighbour distance, normalized coordinates, and cell density.
- Added **multi-channel per-object intensity analysis**.
- Added channel-level intensity/saturation summaries.
- Added basic channel colocalization utilities.
- Added morphological background correction.
- Added dedicated regression tests for quantitative and dimension-aware analysis.

## Core capabilities

### Acquisition QC

Each analyzed image can report:

| Metric | Purpose |
|---|---|
| Focus score | Variance of Laplacian; low values can indicate blur |
| Mean brightness | Detects under/over-exposure |
| Brightness variability | Image contrast / texture proxy |
| Saturation fraction | Detects clipped pixels |
| Dimensions/channels/dtype | Captures acquisition structure |
| SHA-256 | Reproducibility and file identity |

Absolute flags include `BLURRY`, `TOO_DARK`, `TOO_BRIGHT`, `SATURATED`, `FEW_OR_NO_CELLS`, `TOO_MANY_CELLS`, `LOW_SEGMENTATION_QUALITY`, `SEGMENTATION_FALLBACK`, `SEGMENTATION_WARNING`, and `FAILED_TO_LOAD`.

### Segmentation

OptiCell currently exposes:

- `threshold`: Otsu or local adaptive thresholding + morphology + connected components.
- `cellpose`: optional deep-learning backend with a persistent model object per batch.

Every backend returns a common segmentation result containing masks plus diagnostics.

### Segmentation diagnostics

OptiCell measures:

- foreground fraction
- median object area
- coefficient of variation of object area
- border-object fraction
- tiny-object fraction
- merged-object fraction
- segmentation quality score

This catches images where an algorithm returns masks that are technically valid but biologically implausible.

### Cell-level morphology

`extract_object_features()` reports per-object:

- area and perimeter
- circularity
- bounding box
- aspect ratio
- centroid
- mean/std/max intensity

### Spatial phenotyping

`quantitative.py` adds:

- nearest-neighbour distances
- cell density per area
- normalized x/y position
- summary spatial statistics

These are useful for clustering, dispersion, migration, and tissue-architecture analyses.

### Multi-channel quantification

For H×W×C arrays, OptiCell can calculate:

- channel-wise mean/std/min/max
- percentile ranges
- clipping fractions
- per-cell intensity by channel
- integrated intensity
- Pearson-style channel colocalization

### Background correction

A morphology-based rolling background subtraction helper is available for 2-D uint8 images. It is intentionally explicit rather than automatically altering the source image.

### Native C/Z/T microscopy data

`image_io.py` preserves TIFF series axes when the microscope writes them into metadata.

```python
from image_io import load_tiff_stack, canonicalize_axes, project_z, select_channel, select_time

stack = load_tiff_stack("experiment.tif")
stack = canonicalize_axes(stack)
channel = select_channel(stack, 0)
frame = select_time(stack, 3)
projection = project_z(frame, "max")
```

OptiCell no longer needs to pretend every TIFF is a single 2-D image. Ambiguous or unsupported axes are rejected rather than silently reassigned.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e .
```

With Cellpose:

```bash
pip install -e ".[cellpose]"
```

Development tools:

```bash
pip install -e ".[dev]"
```

## CLI

Analyze a directory recursively:

```bash
opticell /path/to/images -o qc_summary.csv
```

Generate CSV plus provenance JSON:

```bash
opticell /path/to/images -o qc_summary.csv --json qc_summary.json
```

Use Cellpose:

```bash
opticell /path/to/images --cell-method cellpose -o qc_summary.csv
```

Use local adaptive segmentation thresholding:

```bash
opticell /path/to/images --adaptive-threshold -o qc_summary.csv
```

Disable dataset-relative outlier scoring when strict absolute-threshold reproducibility is required:

```bash
opticell /path/to/images --no-adaptive-qc -o qc_summary.csv
```

## Python API

```python
from qc_pipeline import QCThresholds, analyze_folder, analyze_image, extract_object_features
from quantitative import add_spatial_features, summarize_spatial_features

thresholds = QCThresholds(focus_min=100, brightness_min=25, brightness_max=230)

df = analyze_folder("./experiment_images", thresholds=thresholds, adaptive_qc=True)

result, segmentation = analyze_image(
    "./experiment_images/sample_01.tif",
    thresholds=thresholds,
    return_segmentation=True,
)

features = extract_object_features(gray_image, segmentation.labels)
features = add_spatial_features(features, gray_image.shape)
summary = summarize_spatial_features(features, gray_image.shape)
```

## Output

The primary QC CSV contains fields such as:

- filename/path/hash
- width/height/channels/dtype/ndim
- focus, brightness, contrast, saturation
- estimated cell count + method
- segmentation quality metrics
- cell area statistics
- adaptive QC score and robust-z columns
- flags/errors

JSON export also records pipeline version, requested method, input path, and thresholds.

## Testing

```bash
pytest
python -m compileall -q qc_pipeline.py quantitative.py image_io.py
python qc_pipeline.py --help
```

CI tests Python 3.10, 3.11, and 3.12.

## Project structure

```text
Virelion-OptiCell/
├── qc_pipeline.py                 # Core QC/segmentation engine + CLI
├── quantitative.py                # Spatial + multi-channel quantitative layer
├── image_io.py                    # Dimension-aware TIFF I/O (C/Z/T)
├── make_test_images.py            # Synthetic microscopy fixtures
├── tests/
│   ├── test_qc_pipeline.py
│   └── test_quantitative.py
├── .github/workflows/ci.yml       # Automated CI
├── pyproject.toml                 # Packaging + CLI metadata
├── requirements.txt               # Runtime dependencies
└── LICENSE
```

## Important scientific limitations

OptiCell is **not a validated clinical measurement system** and should not be presented as one.

Segmentation accuracy depends strongly on cell type, staining, modality, magnification, and image quality. Classical thresholding can fail on crowded or heterogeneous fields; Cellpose should be benchmarked on the actual imaging domain.

The dimension-aware I/O layer preserves native stack structure, but the main QC pipeline still operates on a 2-D analysis view unless the caller explicitly selects/projections dimensions first.

Before using OptiCell outputs as experimental endpoints, validate them against a representative ground-truth subset and keep the analysis parameters/provenance with the experiment.
