# Virelion-OptiCell

OptiCell is a headless Python toolkit for microscopy quality control, cell segmentation, tracking, phenotyping, experiment QC, and quantitative image analysis.

## Scope

- acquisition QC for focus, brightness, contrast, saturation, dimensions, channels, clipping, illumination, and artifacts;
- threshold and adaptive-threshold segmentation plus optional Cellpose backends;
- segmentation acceptance and parameter-sensitivity analysis;
- 2-D/3-D tracking and lineage tables;
- morphology, intensity, texture, density, and spatial features;
- nuclear/cytoplasmic compartment measurements;
- time-lapse event analysis;
- plate/well QC, control normalization, edge-effect analysis, Z-prime, and sample-size planning;
- replicate-aware statistics, bootstrap intervals, permutation tests, and FDR;
- TIFF/OME-TIFF I/O and large-image iteration;
- deterministic outputs and input/parameter provenance.

OptiCell has no required GUI or Streamlit dependency. The public Python API is under `opticell`.

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

Development dependencies:

```bash
pip install -e '.[dev]'
```

## CLI

```bash
opticell /path/to/images -o qc_summary.csv
opticell /path/to/images -o qc_summary.csv --json qc_summary.json
opticell /path/to/images --cell-method cellpose -o qc_summary.csv
```

## Python API

```python
from opticell import analyze_folder, extract_object_features

result = analyze_folder("images/")
```

Additional public functions cover segmentation acceptance, artifact metrics, sensitivity analysis, lineage, TIFF memory mapping, profiling, and experiment auditing.

## Validation

Segmentation benchmarking supports pixel/voxel overlap metrics, instance matching, count error, runtime, and failure accounting. Robustness summaries describe sensitivity to analysis parameters but do not establish segmentation correctness. Ground-truth validation remains necessary.

## Scientific limitations

Image-derived measurements depend on acquisition settings, preprocessing, segmentation quality, and biological context. QC gates are decision aids, not evidence of biological validity. Statistical calculations must follow the experimental unit defined by the study design.

## Testing

```bash
pytest
```

CI covers supported Python versions, linting, compilation, CLI checks, distribution builds, and public API imports.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.

## Citation

Cite the repository release and the imaging datasets/experimental methods used for analysis.
