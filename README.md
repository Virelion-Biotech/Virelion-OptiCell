# Virelion-OptiCell

**OptiCell 2.2** is a headless, research-oriented microscopy quality-control and quantitative cell-analysis toolkit from Virelion Biotech.

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
- Added native dimension-aware TIFF I/O with explicit C/Z/T axis handling.
- Added max/mean/median Z projections without silently destroying other axes.
- Added explicit timepoint and channel selection.
- Added spatial phenotype features: nearest-neighbour distance, normalized coordinates, and cell density.
- Added multi-channel per-object intensity analysis.
- Added channel-level intensity/saturation summaries.
- Added basic channel colocalization utilities.
- Added morphology-based background correction.

### 2.2 — objective validation layer
- Added binary IoU, Dice, precision, and recall metrics.
- Added instance-level centroid matching with configurable distance tolerance.
- Added object-level F1 and cell-count error metrics.
- Added multi-image benchmark aggregation.
- Added dedicated regression tests for validation logic.

## Core capabilities

### Acquisition QC

Each analyzed image can report focus, brightness, contrast, saturation, dimensions, dtype, channel count, and SHA-256 identity. Absolute flags include `BLURRY`, `TOO_DARK`, `TOO_BRIGHT`, `SATURATED`, `FEW_OR_NO_CELLS`, `TOO_MANY_CELLS`, `LOW_SEGMENTATION_QUALITY`, `SEGMENTATION_FALLBACK`, `SEGMENTATION_WARNING`, and `FAILED_TO_LOAD`.

### Segmentation

OptiCell exposes two interchangeable backends:

- `threshold`: Otsu or local adaptive thresholding + morphology + connected components.
- `cellpose`: optional deep-learning backend with a persistent model object per batch.

Every backend returns a common segmentation result containing masks plus diagnostics.

### Quantitative cell phenotyping

`extract_object_features()` reports area, perimeter, circularity, bounding box, aspect ratio, centroid, and intensity statistics. `quantitative.py` adds nearest-neighbour distance, cell density, normalized position, channel-wise intensity, integrated intensity, background correction, and basic channel colocalization.

### Native C/Z/T microscopy data

`image_io.py` preserves TIFF series axes when the microscope writes them into metadata and provides explicit selection/projection operations rather than silently flattening dimensions.

```python
from image_io import load_tiff_stack, canonicalize_axes, project_z, select_channel, select_time

stack = canonicalize_axes(load_tiff_stack("experiment.tif"))
channel = select_channel(stack, 0)
frame = select_time(stack, 3)
projection = project_z(frame, "max")
```

### Ground-truth validation

`validation.py` provides objective segmentation metrics for manually annotated or synthetic masks:

```python
from validation import benchmark_segmentation

metrics = benchmark_segmentation(predicted_masks, ground_truth_masks)
print(metrics["pixel_dice_mean"])
print(metrics["instance_f1_mean"])
print(metrics["relative_count_error_mean"])
```

This allows OptiCell to compare classical thresholding, Cellpose, future models, or ensemble methods on the same validation set instead of relying on qualitative screenshots.

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

## Testing

```bash
pytest
python -m compileall -q qc_pipeline.py quantitative.py image_io.py validation.py
python qc_pipeline.py --help
```

CI tests Python 3.10, 3.11, and 3.12.

## Project structure

```text
Virelion-OptiCell/
├── qc_pipeline.py                 # Core QC/segmentation engine + CLI
├── quantitative.py                # Spatial + multi-channel analysis
├── image_io.py                    # Dimension-aware TIFF I/O
├── validation.py                  # Objective segmentation benchmarking
├── make_test_images.py            # Synthetic microscopy fixtures
├── tests/
│   ├── test_qc_pipeline.py
│   ├── test_quantitative.py
│   └── test_validation.py
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Important scientific limitations

OptiCell is **not a validated clinical measurement system** and should not be presented as one.

Segmentation accuracy depends strongly on cell type, staining, modality, magnification, and image quality. Classical thresholding can fail on crowded or heterogeneous fields; Cellpose should be benchmarked on the actual imaging domain.

The main QC pipeline still operates on a 2-D analysis view unless the caller explicitly selects/projections dimensions first. The dimension-aware layer preserves native C/Z/T structure but does not automatically solve 3-D segmentation or tracking.

Validation metrics quantify agreement with a chosen ground truth; they do not establish biological validity. Before using OptiCell outputs as experimental endpoints, validate against a representative annotation set, document acquisition/segmentation settings, and preserve provenance with the experiment.
