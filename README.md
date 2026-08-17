# Virelion-OptiCell

**OptiCell 2.6** is a headless, research-oriented microscopy quality-control and quantitative cell-analysis toolkit from Virelion Biotech.

There is **no Streamlit and no GUI dependency**. The stable public API is under `opticell`; legacy top-level analysis modules remain available where compatibility is useful.

## Core capabilities

- Acquisition QC: focus, brightness, contrast, saturation, dimensions, dtype, channel count, hashing, adaptive MAD outliers.
- Segmentation: threshold/adaptive threshold, persistent Cellpose, model-agnostic backend interface, and ensemble disagreement diagnostics.
- 3-D segmentation baseline: deterministic volumetric threshold segmentation with explicit instance labels and QC summaries.
- Cell phenotyping: morphology, intensity, spatial density, nearest-neighbour distance, texture, multi-channel measurements.
- Compartments: nucleus segmentation/assignment, nucleus-to-cell ratio, cytoplasm, nuclear/cytoplasmic intensity.
- Time-lapse: deterministic 2-D tracking with configurable displacement/gaps and track summaries.
- Experiment metadata: plate/well/timepoint parsing and explicit group annotation.
- Statistics: replicate-aware summaries, Cohen's d, permutation testing, Benjamini-Hochberg FDR.
- Validation: pixel/voxel IoU/Dice/precision/recall, 2-D and 3-D instance matching/F1, count error and benchmark aggregation.
- Native TIFF I/O: explicit C/Z/T axis handling and projections.
- Preprocessing: background, illumination and artifact utilities.
- 3-D volumetric analysis: physical-unit object volume, centroids, bounding boxes, anisotropic surface area, volume fraction, density, and KD-tree nearest-neighbour distances.

## Stable package API

```python
from opticell import analyze_folder, extract_object_features, summarize_volume
from opticell.statistics import summarize_by_replicate, compare_two_groups
from opticell.features import add_spatial_features, object_texture_features
from opticell.io import load_tiff_stack, canonicalize_axes
from opticell.segmentation import segment_threshold, ThresholdSegmenter, CellposeBackend, compare_backends
from opticell.tracking import link_frames
from opticell.experiment import parse_metadata
from opticell.volumetric import volume_features
from opticell.volumetric_segmentation import segment_threshold_3d
```

Example 3-D baseline:

```python
from opticell.volumetric_segmentation import segment_threshold_3d
from opticell.volumetric import summarize_volume

seg = segment_threshold_3d(volume, voxel_size=(2.0, 1.0, 0.5), min_volume_voxels=30)
summary = summarize_volume(seg.labels, voxel_size=(2.0, 1.0, 0.5))
```

Validation accepts either 2-D or 3-D masks:

```python
from opticell import benchmark_segmentation
metrics = benchmark_segmentation([predicted_3d], [truth_3d], max_distance_px=3)
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
python -m compileall -q qc_pipeline.py quantitative.py image_io.py validation.py compartments.py phenotype.py tracking.py ensemble.py experiment.py texture.py preprocessing.py volumetric.py volumetric_segmentation.py opticell
python qc_pipeline.py --help
```

CI tests Python 3.10, 3.11, and 3.12 and validates the stable package namespace plus standard-library import behavior.

## Scientific limitations

OptiCell is **not a validated clinical measurement system**. Segmentation and tracking performance depends on cell type, staining, modality, magnification, acquisition quality, and experimental context. Rule-based phenotyping is transparent but is not a substitute for a validated trained classifier where one is required.

Statistics should be performed at the correct experimental unit. OptiCell provides replicate-aware aggregation specifically to reduce cell-level pseudoreplication, but users must define biological and technical replicates correctly.

The 3-D segmentation layer is a deterministic baseline, not a claim of universal volumetric segmentation accuracy. It should be benchmarked against representative 3-D ground truth before use as a primary endpoint.

Before using OptiCell outputs as experimental endpoints, benchmark segmentation/tracking against representative ground truth, document acquisition and analysis settings, and preserve provenance.
