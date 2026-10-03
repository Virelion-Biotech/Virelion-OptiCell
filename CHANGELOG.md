# Changelog

All notable OptiCell changes are documented here.

## Unreleased

### Added

- Pixel-level instance-mask brush correction in the human-review UI: add to an existing instance, erase pixels, or paint new disconnected objects with auditable corrected-mask persistence.
- Real CUDA + Cellpose + Streamlit UI self-hosted GitHub Actions workflow and runner setup documentation.


## 2.19.0 — 2026-09-27

### Added

- Persistent SQLite-backed users, projects, samples, runs, review decisions, audit logs, quotas, and consistent backups.
- Multipage Streamlit workflows for batch/project analysis, dataset QC, human review, calibration/validation, large-data inspection, model presets, and operations.
- Content-addressed provenance manifest v2 with input/model hashes, package versions, source commit, parameters, runtime, and manifest digest verification.
- Dataset-level robust outlier detection, replicate QC, plate views, and batch run bundles.
- Versioned segmentation presets, configuration fingerprints, custom Cellpose checkpoint persistence, and checkpoint SHA-256 recording.
- Empirical score calibration and threshold sensitivity/specificity/precision analysis from labeled outcomes.
- Validation evidence registry that distinguishes repository benchmarks from independent-lab evidence.
- Memory-conscious OME-TIFF plane iteration planning and pyramid inspection with explicit refusal to silently full-load unsupported giant series.
- Docker/Compose deployment, optional local authentication, health checks, audit/backup UI, and release automation.
- Scheduled/manual real Cellpose smoke workflow plus injected Cellpose contract coverage in normal tests.

### Scientific scope

- The new calibration tooling does not make existing QC scores calibrated probabilities without labeled domain-specific data.
- No independent-laboratory dataset was invented or reclassified; external validation remains an evidence-generation task.
- Corrected masks can be attached and audited, but pixel-level browser mask painting is not yet implemented.


## 2.18.0 — 2026-09-20

### Added

- Phase/low-contrast-aware automatic backend selection on the production segmentation path.
- Streamlit AppTest smoke coverage and end-to-end killer-workflow regression coverage.
- Optional HITL review queue with auditable accept/reject/rerun decisions.
- Single-source package versioning, SPDX licensing, and dependency/security release checks.
- Current T4 LIVECell n=20 evidence for Cellpose, hybrid, auto, and threshold paths.

### Documentation

- Added `DATASETS_AND_VALIDATION_REPORT.md`: catalogue of real large-scale public microscopy datasets (LIVECell, TissueNet, BBBC focus/QC sets, Cell Painting collections, etc.) suitable for OptiCell validation and future training, plus an explicit no-hallucination policy. No training metrics are claimed; full-scale training requires data + compute outside the present environment.

## 2.16.0 — 2026-08-18

### Added

- Experiment-level audit combining acquisition QC, segmentation QC, reproducibility fingerprints, and manifest comparisons.
- Deterministic analysis fingerprints and environment metadata.
- Screening statistics including robust normalization, SSMD, Z-prime, and edge-effect summaries.
- Lineage graphs, lineage event diagnostics, and tracking-quality metrics.
- OME-TIFF metadata parsing and memory-safe TIFF streaming utilities.
- Segmentation sensitivity/robustness analysis and explicit acceptance gates.
- Artifact-quality scoring and composite experiment quality gates.
- Reproducible benchmark, profiling, provenance, and reporting layers.
- Parallel batch processing and pluggable segmentation backends.
- 2-D and 3-D tracking with physical-unit distances and assignment-based matching.

### Quality and packaging

- Headless API/CLI architecture with Streamlit removed.
- Python 3.10–3.12 CI coverage.
- Correctness-focused linting, compilation, CLI smoke tests, and distribution builds.
- Public API contract and built-artifact import validation.
- Isolated wheel/sdist installation and dependency checks in CI.

### Scientific scope

OptiCell remains a research analysis toolkit. Segmentation, tracking, lineage, QC thresholds, screening statistics, and power calculations require dataset-specific validation and should not be interpreted as clinical or universally validated measurements.
