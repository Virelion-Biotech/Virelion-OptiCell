# Virelion-OptiCell

**OptiCell 2.4** is a headless, research-oriented microscopy quality-control and quantitative cell-analysis toolkit from Virelion Biotech.

There is **no Streamlit and no GUI dependency**. OptiCell is a reusable Python package plus CLI for local analysis, notebooks, servers, CI, and larger bioinformatics pipelines. Legacy top-level modules remain available for compatibility, while the stable `opticell.*` namespace is now the preferred API.

## Major capabilities

### Acquisition and segmentation QC
- Focus, brightness, contrast, saturation, dimensions, dtype, channel count, and SHA-256 identity.
- Absolute QC flags plus dataset-relative robust MAD outlier scoring.
- Threshold and adaptive threshold segmentation.
- Optional persistent Cellpose backend.
- Segmentation-quality diagnostics and disagreement-aware ensemble analysis.

### Quantitative cell phenotyping
- Per-cell morphology and intensity features.
- Spatial statistics: nearest-neighbour distances, density, normalized position.
- Multi-channel per-cell intensity and integrated intensity.
- Texture and heterogeneity descriptors.
- Explicit background, denoising, flat-field, and artifact preprocessing utilities.

### Nucleus/cell compartments
`compartments.py` adds:
- nucleus-to-cell area ratio
- cytoplasm area
- nucleus/cytoplasm intensity ratio
- nucleus assignment to parent cells
- multiple-nucleus detection per cell

### Auditable phenotype classification
`phenotype.py` provides rule-based phenotype scoring where every classification records contributing rules, score, and score fraction rather than hiding decisions inside a black box.

### Time-lapse tracking
`tracking.py` provides deterministic 2-D centroid tracking with configurable displacement and short-gap handling plus trajectory summaries.

### Segmentation ensembles
`ensemble.py` can compare threshold strategies or precomputed model outputs and surfaces disagreement rather than hiding uncertainty.

### Plate / experiment metadata
`experiment.py` supports plate/well/timepoint extraction, explicit metadata injection, and experiment summaries.

### Replicate-aware statistics
`statistics.py` is designed around the **experimental unit**, not individual cells. It provides:
- aggregation to biological/technical replicate level
- replicate-level group summaries
- Cohen's d
- permutation-based two-group testing
- Benjamini-Hochberg FDR adjustment

This is intentionally separate from cell-level feature extraction so users are less likely to commit pseudoreplication errors.

### Native C/Z/T microscopy I/O
`image_io.py` preserves TIFF dimensions and provides explicit C/Z/T selection and projection rather than silently flattening microscopy data.

### Objective validation
`validation.py` provides pixel IoU/Dice, precision/recall, instance matching, F1, cell-count error, and multi-image benchmark aggregation against ground truth masks.

## Package layout

```text
opticell/
├── __init__.py
├── analysis.py
├── features.py
├── io.py
├── segmentation.py
├── statistics.py
└── tracking.py
```

The legacy modules (`qc_pipeline.py`, `quantitative.py`, `image_io.py`, etc.) remain importable to avoid breaking existing scripts.

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

## Preferred Python API

```python
import opticell
from opticell.features import extract_object_features, add_spatial_features, object_texture_features
from opticell.segmentation import segment_threshold, CellposeSegmenter
from opticell.analysis import segment_nuclei, compartment_features, Rule, score_cells
from opticell.io import load_tiff_stack, canonicalize_axes, project_z
from opticell.statistics import summarize_by_replicate, compare_two_groups

qc = opticell.analyze_folder("./images", adaptive_qc=True)
result, segmentation = opticell.analyze_image("./images/sample.tif", return_segmentation=True)
features = extract_object_features(gray_image, segmentation.labels)
features = add_spatial_features(features, gray_image.shape)
features = features.join(object_texture_features(gray_image, segmentation.labels).set_index("label"), on="label", rsuffix="_texture")

stack = canonicalize_axes(load_tiff_stack("experiment.tif"))
projection = project_z(stack, "max")
```

## CLI

```bash
opticell /path/to/images -o qc_summary.csv
opticell /path/to/images -o qc_summary.csv --json qc_summary.json
opticell /path/to/images --cell-method cellpose -o qc_summary.csv
opticell /path/to/images --adaptive-threshold -o qc_summary.csv
opticell /path/to/images --no-adaptive-qc -o qc_summary.csv
```

## Testing

```bash
pytest
python -m compileall -q qc_pipeline.py quantitative.py image_io.py validation.py compartments.py phenotype.py tracking.py ensemble.py experiment.py statistics.py texture.py preprocessing.py opticell
python qc_pipeline.py --help
```

CI tests Python 3.10, 3.11, and 3.12 and imports the public `opticell` package.

## Scientific limitations

OptiCell is **not a validated clinical measurement system**. Segmentation and tracking performance depends on cell type, staining, modality, magnification, acquisition quality, and experimental context. Rule-based phenotyping is transparent but is not a substitute for a validated trained classifier when one is required.

Most importantly, cell-level sample size is not automatically biological sample size. Use the replicate-aware statistics layer and define the true experimental unit before inferential testing. Benchmark segmentation/tracking against representative ground truth and preserve the analysis provenance with the experiment.
