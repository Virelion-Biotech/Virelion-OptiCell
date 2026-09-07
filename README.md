# Virelion-OptiCell

OptiCell is a headless Python toolkit for microscopy quality control, cell segmentation, tracking, phenotyping, experiment QC, and quantitative image analysis.

## What it contains

- Acquisition QC for focus, brightness, contrast, saturation, dimensions, channels, clipping, illumination, and artifacts.
- Threshold and adaptive-threshold segmentation with optional Cellpose backends.
- Segmentation acceptance and parameter-sensitivity analysis.
- 2-D/3-D tracking and lineage tables.
- Morphology, intensity, texture, density, and spatial features.
- Nuclear/cytoplasmic compartment measurements.
- Time-lapse event analysis.
- Plate/well QC, control normalization, edge-effect analysis, Z-prime, and sample-size planning.
- Replicate-aware statistics, bootstrap intervals, permutation tests, and FDR.
- TIFF/OME-TIFF I/O and large-image iteration.
- Deterministic outputs and input/parameter provenance.

There is no required GUI or Streamlit dependency. The public Python API is under `opticell`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Optional Cellpose support:

```bash
pip install -e '.[cellpose]'
```

## Usage

```bash
opticell /path/to/images -o qc_summary.csv
opticell /path/to/images -o qc_summary.csv --json qc_summary.json
opticell /path/to/images --cell-method cellpose -o qc_summary.csv
```

Python:

```python
from opticell import analyze_folder

result = analyze_folder("images/")
```

## Inputs and outputs

**Inputs:** TIFF/OME-TIFF microscopy images or image folders, acquisition metadata, segmentation/tracking parameters, optional masks/labels, and experimental grouping information.

**Outputs:** QC summaries, segmentation results, object-level feature tables, tracks/lineages, time-lapse events, experiment-level statistics, and provenance records.

## Validation

Segmentation benchmarking supports pixel/voxel overlap metrics, instance matching, count error, runtime, and failure accounting. Robustness summaries measure sensitivity to analysis parameters. Software tests and CI cover supported Python versions, linting, compilation, CLI checks, distribution builds, and public API imports.

Ground-truth image annotations remain necessary to establish segmentation correctness.

## Limitations

Image-derived measurements depend on acquisition settings, preprocessing, segmentation quality, and biological context. QC gates are decision aids, not evidence of biological validity. Statistical calculations must use the experimental unit defined by the study design. Cellpose performance depends on the selected model and image domain.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.
