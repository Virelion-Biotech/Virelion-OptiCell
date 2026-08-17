# Virelion-OptiCell

**OptiCell 2.3** is a headless, research-oriented microscopy quality-control and quantitative cell-analysis toolkit from Virelion Biotech.

There is **no Streamlit and no GUI dependency**. OptiCell is designed as a reusable Python API plus CLI for local analysis, notebooks, servers, CI, and larger bioinformatics pipelines.

## Major capabilities

### Acquisition and segmentation QC
- Focus, brightness, contrast, saturation, dimensions, dtype, channel count, and SHA-256 identity.
- Absolute QC flags plus dataset-relative robust MAD outlier scoring.
- Threshold and adaptive threshold segmentation.
- Optional persistent Cellpose backend.
- Segmentation-quality diagnostics: foreground fraction, area statistics, border/tiny/merged-object rates, and quality score.

### Quantitative cell phenotyping
- Per-cell morphology and intensity features.
- Spatial statistics: nearest-neighbour distances, density, normalized position.
- Multi-channel per-cell intensity and integrated intensity.
- Background correction and channel colocalization.

### Nucleus/cell compartments
`compartments.py` adds explicit nucleus-aware measurements:
- nucleus-to-cell area ratio
- cytoplasm area
- nucleus/cytoplasm intensity ratio
- nucleus assignment to parent cells
- multiple-nucleus detection per cell

This provides a foundation for nuclear translocation, differentiation, toxicity, and subcellular localization analyses.

### Auditable phenotype classification
`phenotype.py` provides rule-based phenotype scoring. Every classification records the contributing rules, score, and score fraction rather than hiding decisions inside a black box.

```python
from phenotype import Rule, score_cells
rules = [
    Rule("mean_intensity", 80, label="marker_positive"),
    Rule("circularity", 0.7, label="round_cell"),
]
scored = score_cells(features, rules)
```

### Time-lapse tracking
`tracking.py` provides deterministic centroid-based tracking for 2-D time series:
- one-to-one nearest-neighbour linking
- configurable maximum displacement
- optional short-gap handling
- per-track path length
- net displacement
- mean and net speed

It is deliberately lightweight; domain-specific tracking should still be validated against annotated tracks.

### Segmentation ensembles
`ensemble.py` runs multiple threshold strategies or combines precomputed segmentations and exposes model disagreement instead of hiding it.

```python
from ensemble import threshold_ensemble
result = threshold_ensemble(gray_image)
print(result.member_counts)
print(result.agreement_fraction)
print(result.warning)
```

Large disagreement is surfaced as `SEGMENTATION_DISAGREEMENT`, making uncertain images obvious.

### Plate / experiment metadata
`experiment.py` adds:
- plate extraction
- well extraction
- timepoint parsing
- explicit user metadata injection
- group summaries
- 8×12-style plate heatmap matrices

OptiCell does not infer biological condition labels from arbitrary filenames; conditions should be supplied explicitly.

### Native C/Z/T microscopy I/O
`image_io.py` preserves TIFF dimensions and provides explicit C/Z/T selection and projection rather than silently flattening microscopy data.

### Objective validation
`validation.py` provides pixel IoU/Dice, precision/recall, instance matching, F1, cell-count error, and multi-image benchmark aggregation against ground truth masks.

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

```bash
opticell /path/to/images -o qc_summary.csv
opticell /path/to/images -o qc_summary.csv --json qc_summary.json
opticell /path/to/images --cell-method cellpose -o qc_summary.csv
opticell /path/to/images --adaptive-threshold -o qc_summary.csv
opticell /path/to/images --no-adaptive-qc -o qc_summary.csv
```

## Python API

```python
from qc_pipeline import analyze_folder, analyze_image, extract_object_features
from quantitative import add_spatial_features
from compartments import segment_nuclei, compartment_features
from phenotype import Rule, score_cells
from tracking import link_frames, summarize_tracks
from experiment import annotate_results

qc = analyze_folder("./images", adaptive_qc=True)
result, segmentation = analyze_image("./images/sample.tif", return_segmentation=True)
features = extract_object_features(gray_image, segmentation.labels)
features = add_spatial_features(features, gray_image.shape)

nuclei = segment_nuclei(dapi_image)
compartments = compartment_features(marker_image, segmentation.labels, nuclei)
scored = score_cells(features, [Rule("mean_intensity", 80, label="marker_positive")])

tracks = link_frames(timepoint_labels)
motion = summarize_tracks(tracks)
experiment = annotate_results(qc)
```

## Testing

```bash
pytest
python -m compileall -q qc_pipeline.py quantitative.py image_io.py validation.py compartments.py phenotype.py tracking.py ensemble.py experiment.py
python qc_pipeline.py --help
```

CI tests Python 3.10, 3.11, and 3.12.

## Project structure

```text
Virelion-OptiCell/
├── qc_pipeline.py
├── quantitative.py
├── image_io.py
├── validation.py
├── compartments.py
├── phenotype.py
├── tracking.py
├── ensemble.py
├── experiment.py
├── make_test_images.py
├── tests/
│   ├── test_qc_pipeline.py
│   ├── test_quantitative.py
│   ├── test_validation.py
│   └── test_advanced.py
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Scientific limitations

OptiCell is **not a validated clinical measurement system**. Segmentation and tracking performance depends on cell type, staining, modality, magnification, acquisition quality, and experimental context. Rule-based phenotyping is intentionally transparent but is not a substitute for a trained classifier when a validated model is required.

Before using OptiCell outputs as experimental endpoints, benchmark segmentation/tracking against representative ground truth, define the experimental unit and replicates correctly, document acquisition and analysis settings, and preserve the generated provenance.
