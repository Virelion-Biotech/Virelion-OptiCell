# Virelion-OptiCell

**OptiCell 2.16** is a headless, research-oriented microscopy quality-control and quantitative cell-analysis toolkit from Virelion Biotech.

There is **no Streamlit and no GUI dependency**. The stable public API is under `opticell`; legacy top-level analysis modules remain available where compatibility is useful.

## Core capabilities

- Acquisition QC: focus, brightness, contrast, saturation, dimensions, dtype, channel count, hashing, adaptive MAD outliers, clipping/hot-pixel detection, illumination-gradient diagnostics, and artifact-burden scoring.
- Segmentation: threshold/adaptive threshold, persistent Cellpose, CPU/GPU selection when supported, model-agnostic backend interface, custom registry, ensemble disagreement diagnostics, and parameter-sensitivity analysis.
- Segmentation acceptance: transparent PASS/REVIEW/FAIL gates using quality, border, tiny-object, merged-object, and optional ensemble-agreement thresholds.
- Segmentation robustness: aggregate sensitivity stability, coefficient-of-variation summaries, and stable-parameter subsets without equating stability with correctness.
- Segmentation benchmarking: ground-truth comparisons with runtime, pixel/voxel metrics, instance metrics, count error, and failure accounting.
- High-throughput batch analysis: parallel workers, deterministic output ordering, progress callbacks, and failure isolation.
- 2-D/3-D tracking: one-to-one assignment, short gaps, velocity prediction, physical-unit 3-D matching, confidence, trajectory speed, path length, and straightness.
- Lineage analysis: auditable parent/child relationships, track-continuity/fragmentation diagnostics, and split/merge/appearance/disappearance event-rate and division-consistency metrics.
- 3-D segmentation baseline: deterministic volumetric threshold segmentation with explicit instance labels and QC summaries.
- Cell phenotyping: morphology, intensity, spatial density, nearest-neighbour distance, texture, and multi-channel measurements.
- Compartments: nucleus segmentation/assignment, nucleus-to-cell ratio, cytoplasm, and nuclear/cytoplasmic intensity.
- Time-lapse event analysis: split, merge, appearance, disappearance, and division-event tables.
- Experiment metadata/QC: plate/well/timepoint parsing, explicit group annotation, control normalization, plate edge-effect analysis, Z′, prospective two-group sample-size planning, and auditable assay pass/marginal/fail decisions.
- Statistics: replicate-aware summaries, bootstrap confidence intervals, Cohen's d, permutation testing, and Benjamini-Hochberg FDR.
- Validation: pixel/voxel IoU/Dice/precision/recall, 2-D/3-D instance matching/F1, sparse-label-safe counts, globally optimal one-to-one centroid matching, count error, and benchmark aggregation.
- Native TIFF/OME-TIFF I/O: explicit C/Z/T axis handling, physical scales, units, channel metadata, projections, and series selection.
- Streaming I/O: memory mapping where supported and frame/chunk iteration for large TIFF datasets.
- Export interoperability: deterministic long-form feature tables plus CSV/Parquet output when a Parquet engine is installed.
- Preprocessing: background, illumination, and artifact utilities.
- 3-D volumetric analysis: physical-unit object volume, centroids, bounding boxes, anisotropic surface area, volume fraction, density, and KD-tree nearest-neighbour distances.
- Reproducibility: input SHA-256 hashes, runtime/platform metadata, deterministic analysis fingerprints, parameter manifests, and manifest comparison.
- Experiment auditing: a machine-readable composite audit combining acquisition QC, segmentation QC, reproducibility fingerprints, and reference-manifest differences without hiding component evidence.
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
    assay_qc_decision,
    segmentation_acceptance,
    acquisition_artifact_metrics,
    artifact_burden_score,
    threshold_sensitivity,
    summarize_sensitivity,
    stable_parameter_subset,
    build_lineage_table,
    lineage_event_summary,
    memmap_tiff,
    profile_call,
    audit_experiment,
)
```

### Experiment audit

```python
from opticell import audit_experiment

audit = audit_experiment(
    parameters={"threshold": 0.5},
    input_hashes={"image": "<sha256>"},
    artifact_score=92,
    segmentation_score=89,
    artifact_status="PASS",
    segmentation_status="PASS",
)
print(audit.status, audit.score, audit.fingerprint)
```

The audit is deliberately transparent: it reports the composite gate while preserving the underlying QC reasons and, when supplied, whether inputs and parameters match a reference manifest. It is a reproducibility/QC aid, not proof of biological validity.

### Assay QC decision

```python
from opticell import z_prime_factor, assay_qc_decision

zprime = z_prime_factor(negative_controls, positive_controls)
decision = assay_qc_decision(zprime)
print(decision["status"], decision["reason"])
```

The decision layer is an explicit screening/QC rule; it does not establish biological validity.

### Acquisition artifact metrics

```python
from opticell import acquisition_artifact_metrics, artifact_burden_score

metrics = acquisition_artifact_metrics(image)
score = artifact_burden_score(metrics)
```

These are descriptive acquisition/QC signals. They are not automatic claims of microscope failure or biological abnormality.

### Segmentation acceptance and robustness

```python
from opticell import segmentation_acceptance, summarize_sensitivity

acceptance = segmentation_acceptance(
    quality_score=segmentation.quality_score,
    border_fraction=segmentation.border_fraction,
    tiny_object_fraction=segmentation.tiny_object_fraction,
    merged_object_fraction=segmentation.merged_object_fraction,
)

robustness = summarize_sensitivity(sensitivity_table, value_column="object_count")
```

Acceptance rules and robustness summaries are transparent decision aids. Stable results do not imply correct segmentation, so representative ground-truth validation remains necessary.

### Segmentation sensitivity

```python
from opticell import threshold_sensitivity

summary = threshold_sensitivity(
    image,
    thresholds=[80, 100, 120, 140],
    segmenter=my_threshold_segmenter,
)
print(summary[["threshold", "object_count", "count_cv"]])
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

The CI matrix validates Python 3.10, 3.11, and 3.12 with pytest, correctness-focused Ruff checks, module compilation, CLI help, distribution builds, and public API import smoke tests.
