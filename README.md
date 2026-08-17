# Virelion-OptiCell

**OptiCell 2.12** is a headless, research-oriented microscopy quality-control and quantitative cell-analysis toolkit from Virelion Biotech.

There is **no Streamlit and no GUI dependency**. The stable public API is under `opticell`; legacy top-level analysis modules remain available where compatibility is useful.

## Core capabilities

- Acquisition QC: focus, brightness, contrast, saturation, dimensions, dtype, channel count, hashing, adaptive MAD outliers.
- Segmentation: threshold/adaptive threshold, persistent Cellpose, CPU/GPU selection when supported, model-agnostic backend interface, custom registry, and ensemble disagreement diagnostics.
- Segmentation benchmarking: ground-truth comparisons with runtime, pixel/voxel metrics, instance metrics, count error, and failure accounting.
- High-throughput batch analysis: parallel workers, deterministic output ordering, progress callbacks, and failure isolation.
- 2-D/3-D tracking: one-to-one assignment, short gaps, velocity prediction, physical-unit 3-D matching, confidence, trajectory speed, path length, and straightness.
- Lineage analysis: auditable parent/child relationships plus track-continuity and fragmentation diagnostics.
- 3-D segmentation baseline: deterministic volumetric threshold segmentation with explicit instance labels and QC summaries.
- Cell phenotyping: morphology, intensity, spatial density, nearest-neighbour distance, texture, and multi-channel measurements.
- Compartments: nucleus segmentation/assignment, nucleus-to-cell ratio, cytoplasm, and nuclear/cytoplasmic intensity.
- Time-lapse event analysis: split, merge, appearance, disappearance, and division-event tables.
- Experiment metadata/QC: plate/well/timepoint parsing, explicit group annotation, control normalization, plate edge-effect analysis, Z′, and prospective two-group sample-size planning.
- Statistics: replicate-aware summaries, bootstrap confidence intervals, Cohen's d, permutation testing, and Benjamini-Hochberg FDR.
- Validation: pixel/voxel IoU/Dice/precision/recall, 2-D/3-D instance matching/F1, count error, and benchmark aggregation.
- Native TIFF/OME-TIFF I/O: explicit C/Z/T axis handling, physical scales, units, channel metadata, projections, and series selection.
- Streaming I/O: memory mapping where supported and frame/chunk iteration for large TIFF datasets.
- Export interoperability: deterministic long-form feature tables plus CSV/Parquet output when a Parquet engine is installed.
- Preprocessing: background, illumination, and artifact utilities.
- 3-D volumetric analysis: physical-unit object volume, centroids, bounding boxes, anisotropic surface area, volume fraction, density, and KD-tree nearest-neighbour distances.
- Reproducibility: input SHA-256 hashes, runtime/platform metadata, parameter manifests, and JSON reports.
- Performance profiling: structured operation timing, throughput, and machine-readable profiling tables.

## Stable package API

```python
from opticell import (
    analyze_folder,
    extract_object_features,
    summarize_volume,
    normalize_to_controls,
    plate_edge_effect,
    z_prime_factor,
    two_group_sample_size,
    build_lineage_table,
    lineage_quality_summary,
    read_ome_info,
    dataframe_to_long_form,
    memmap_tiff,
    profile_call,
)
```

### OME-TIFF metadata

```python
from opticell import read_ome_info

info = read_ome_info("sample.ome.tif")
print(info.axes, info.shape, info.channel_names)
print(info.physical_size_x, info.physical_size_y, info.physical_size_z)
```

### Screening and prospective planning

```python
from opticell import normalize_to_controls, z_prime_factor, two_group_sample_size

normalized = normalize_to_controls(
    results,
    value_column="cell_count",
    control_column="condition",
    control_value="control",
)

zprime = z_prime_factor(negative_controls, positive_controls)
plan = two_group_sample_size(effect_size=0.8, alpha=0.05, power=0.8)
```

The power helper is a normal-approximation planning tool; clustered designs, blocking, repeated measures, and other complex experiments require design-specific power analysis.

### Lineage quality

```python
from opticell import lineage_quality_summary

quality = lineage_quality_summary(tracks)
print(quality["track_completeness"], quality["fragmented_tracks"])
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
ruff check . --select E9,F
python -m compileall -q qc_pipeline.py quantitative.py image_io.py validation.py compartments.py phenotype.py tracking.py ensemble.py experiment.py texture.py preprocessing.py volumetric.py volumetric_segmentation.py tracking3d.py provenance.py reporting.py benchmarking.py runtime.py tracking_events.py experiment_stats.py experiment_qc.py profiling.py lineage.py screening.py stream_io.py opticell
python qc_pipeline.py --help
python -m build
```

CI tests Python 3.10, 3.11, and 3.12, runs the full test suite, correctness-focused linting, package compilation/build checks, and public API import validation.

## Scientific limitations

OptiCell is **not a validated clinical measurement system**. Segmentation and tracking performance depends on cell type, staining, modality, magnification, acquisition quality, and experimental context. Rule-based phenotyping is transparent but is not a substitute for a validated trained classifier where one is required.

Statistics should be performed at the correct experimental unit. OptiCell provides replicate-aware aggregation specifically to reduce cell-level pseudoreplication, but users must define biological and technical replicates correctly.

The 3-D segmentation/tracking and lineage layers are transparent baselines, not claims of universal volumetric or lineage accuracy. They should be benchmarked against representative ground truth before use as primary endpoints.

Control-normalized plate metrics assume the specified control group is an appropriate reference. Z′ and edge-effect metrics are screening/QC statistics, not proof of biological mechanism or acquisition failure.

The prospective power calculation is an approximation and should not be used as a substitute for design-specific power analysis when clustering, blocking, repeated measures, non-Gaussian outcomes, or multiple endpoints matter.

OME-TIFF metadata are parsed conservatively from the first TIFF series. Unsupported microscopy container features should not be assumed to be preserved unless explicitly represented by the returned metadata or axes.

Before using OptiCell outputs as experimental endpoints, benchmark segmentation/tracking against representative ground truth, document acquisition and analysis settings, define the experimental unit, and preserve provenance.
