# Virelion-OptiCell

**OptiCell 2.10** is a headless, research-oriented microscopy quality-control and quantitative cell-analysis toolkit from Virelion Biotech.

There is **no Streamlit and no GUI dependency**. The stable public API is under `opticell`; legacy top-level analysis modules remain available where compatibility is useful.

## Core capabilities

- Acquisition QC: focus, brightness, contrast, saturation, dimensions, dtype, channel count, hashing, adaptive MAD outliers.
- Segmentation: threshold/adaptive threshold, persistent Cellpose, automatic CPU/GPU selection when supported, model-agnostic backend interface, custom backend registry, and ensemble disagreement diagnostics.
- Segmentation benchmarking: reproducible backend comparisons against representative ground-truth masks with runtime, pixel/voxel metrics, instance metrics, count error, user metadata, and failure accounting.
- High-throughput batch analysis: parallel workers, deterministic output ordering, progress callbacks, and failure isolation.
- 2-D tracking: one-to-one assignment, optional short-gap handling, velocity prediction, match confidence, trajectory speed, path length, straightness, and gap summaries.
- 3-D tracking: physical-unit one-to-one assignment with anisotropic voxel spacing, velocity prediction, match confidence, and trajectory summaries.
- 3-D segmentation baseline: deterministic volumetric threshold segmentation with explicit instance labels and QC summaries.
- Cell phenotyping: morphology, intensity, spatial density, nearest-neighbour distance, texture, multi-channel measurements.
- Compartments: nucleus segmentation/assignment, nucleus-to-cell ratio, cytoplasm, nuclear/cytoplasmic intensity.
- Time-lapse event analysis: split, merge, appearance, disappearance, and auditable division-event tables.
- Experiment metadata: plate/well/timepoint parsing and explicit group annotation.
- Experiment QC: robust control normalization, plate-level summaries, and edge-effect detection.
- Statistics: replicate-aware summaries, bootstrap confidence intervals, Cohen's d, permutation testing, Benjamini-Hochberg FDR.
- Validation: pixel/voxel IoU/Dice/precision/recall, 2-D and 3-D instance matching/F1, count error and benchmark aggregation.
- Native TIFF I/O: explicit C/Z/T axis handling and projections.
- Preprocessing: background, illumination and artifact utilities.
- 3-D volumetric analysis: physical-unit object volume, centroids, bounding boxes, anisotropic surface area, volume fraction, density, and KD-tree nearest-neighbour distances.
- Reproducibility: input SHA-256 hashes, runtime/platform metadata, parameter manifests, and portable JSON reports.
- Performance profiling: structured per-operation timing, throughput, and machine-readable profiling tables.

## Stable package API

```python
from opticell import (
    analyze_folder,
    extract_object_features,
    summarize_volume,
    normalize_to_controls,
    plate_edge_effect,
    benchmark_backends,
    profile_call,
)
from opticell.statistics import summarize_by_replicate, compare_two_groups
from opticell.features import add_spatial_features, object_texture_features
from opticell.io import load_tiff_stack, canonicalize_axes
from opticell.segmentation import ThresholdSegmenter, CellposeBackend, compare_backends, get_backend
from opticell.tracking3d import Tracking3DConfig, link_frames_3d, summarize_tracks_3d
from opticell.volumetric import volume_features
from opticell.volumetric_segmentation import segment_threshold_3d
```

Example 3-D segmentation + tracking:

```python
from opticell.volumetric_segmentation import segment_threshold_3d
from opticell.tracking3d import Tracking3DConfig, link_frames_3d

segments = [
    segment_threshold_3d(volume, voxel_size=(2.0, 1.0, 0.5), min_volume_voxels=30).labels
    for volume in time_volumes
]
tracks = link_frames_3d(
    segments,
    voxel_size=(2.0, 1.0, 0.5),
    frame_interval=5.0,
    config=Tracking3DConfig(max_distance_um=20, max_gap=1),
)
```

Example control normalization:

```python
from opticell import normalize_to_controls

normalized = normalize_to_controls(
    results,
    metric="cell_count",
    group_column="condition",
    control_value="control",
)
```

Example profiling:

```python
from opticell import profile_call

result, timing = profile_call("segmentation", segmenter.segment, image, items=1)
print(timing.items_per_second)
```

Example provenance:

```python
from opticell import build_manifest, write_manifest

manifest = build_manifest(
    opticell_version="2.10.0",
    inputs=image_paths,
    parameters={"cell_method": "cellpose", "workers": 4},
    operation="batch_qc",
)
write_manifest(manifest, "results/provenance.json")
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
ruff check . --select E4,E7,E9,F
python -m compileall -q qc_pipeline.py quantitative.py image_io.py validation.py compartments.py phenotype.py tracking.py ensemble.py experiment.py texture.py preprocessing.py volumetric.py volumetric_segmentation.py tracking3d.py provenance.py reporting.py benchmarking.py runtime.py tracking_events.py experiment_stats.py experiment_qc.py profiling.py opticell
python qc_pipeline.py --help
python -m build
```

CI tests Python 3.10, 3.11, and 3.12, runs tests and correctness-focused linting, validates the stable package namespace, checks standard-library import behavior, and builds the distributable package.

## Scientific limitations

OptiCell is **not a validated clinical measurement system**. Segmentation and tracking performance depends on cell type, staining, modality, magnification, acquisition quality, and experimental context. Rule-based phenotyping is transparent but is not a substitute for a validated trained classifier where one is required.

Statistics should be performed at the correct experimental unit. OptiCell provides replicate-aware aggregation specifically to reduce cell-level pseudoreplication, but users must define biological and technical replicates correctly.

The 3-D segmentation and tracking layers are transparent baselines, not claims of universal volumetric or lineage-tracking accuracy. They should be benchmarked against representative ground truth before use as primary endpoints.

Control-normalized plate metrics assume the specified control group is an appropriate reference. Edge-effect detection is descriptive and should not be treated as proof of an acquisition artifact without experimental review.

Before using OptiCell outputs as experimental endpoints, benchmark segmentation/tracking against representative ground truth, document acquisition and analysis settings, define the experimental unit, and preserve provenance.
