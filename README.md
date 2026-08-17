# Virelion-OptiCell

**OptiCell 2.5** is a headless, research-oriented microscopy quality-control and quantitative cell-analysis toolkit from Virelion Biotech.

There is **no Streamlit and no GUI dependency**. The stable public API is under `opticell`; legacy top-level analysis modules remain available where compatibility is useful.

## Core capabilities

- Acquisition QC: focus, brightness, contrast, saturation, dimensions, dtype, channel count, hashing, adaptive MAD outliers.
- Segmentation: threshold/adaptive threshold, optional persistent Cellpose, ensemble disagreement diagnostics.
- Cell phenotyping: morphology, intensity, spatial density, nearest-neighbour distance, texture, multi-channel measurements.
- Compartments: nucleus segmentation/assignment, nucleus-to-cell ratio, cytoplasm, nuclear/cytoplasmic intensity.
- Time-lapse: deterministic 2-D tracking with configurable displacement/gaps and track summaries.
- Experiment metadata: plate/well/timepoint parsing and explicit group annotation.
- Statistics: replicate-aware summaries, Cohen's d, permutation testing, Benjamini-Hochberg FDR.
- Validation: pixel IoU/Dice/precision/recall, instance matching/F1, count error and benchmark aggregation.
- Native TIFF I/O: explicit C/Z/T axis handling and projections.
- Preprocessing: background, illumination and artifact utilities.
- **3-D volumetric analysis:** physical-unit object volume, centroids, bounding boxes, approximate surface area, volume fraction, density, and 3-D nearest-neighbour distances.

## Stable package API

```python
from opticell import analyze_folder, extract_object_features, summarize_volume
from opticell.statistics import summarize_by_replicate, compare_two_groups
from opticell.features import add_spatial_features, object_texture_features
from opticell.io import load_tiff_stack, canonicalize_axes
from opticell.segmentation import segment_threshold, threshold_ensemble
from opticell.tracking import link_frames
from opticell.experiment import parse_metadata
from opticell.volumetric import volume_features
```

For a labelled 3-D mask:

```python
from opticell.volumetric import volume_features, summarize_volume

features = volume_features(labels, voxel_size=(2.0, 1.0, 0.5))
summary = summarize_volume(labels, voxel_size=(2.0, 1.0, 0.5))
```

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

## Testing

```bash
pytest
python -m compileall -q qc_pipeline.py quantitative.py image_io.py validation.py compartments.py phenotype.py tracking.py ensemble.py experiment.py texture.py preprocessing.py volumetric.py opticell
python qc_pipeline.py --help
```

CI tests Python 3.10, 3.11, and 3.12 and validates the stable package namespace plus standard-library import behavior.

## Scientific limitations

OptiCell is **not a validated clinical measurement system**. Segmentation and tracking performance depends on cell type, staining, modality, magnification, acquisition quality, and experimental context. Rule-based phenotyping is transparent but is not a substitute for a validated trained classifier where one is required.

Statistics should be performed at the correct experimental unit. OptiCell provides replicate-aware aggregation specifically to reduce cell-level pseudoreplication, but users must define biological and technical replicates correctly.

The current 3-D layer quantifies labelled volumes but does not yet provide a validated 3-D segmentation model or full 3-D time-lapse tracker.

Before using OptiCell outputs as experimental endpoints, benchmark segmentation/tracking against representative ground truth, document acquisition and analysis settings, and preserve provenance.
