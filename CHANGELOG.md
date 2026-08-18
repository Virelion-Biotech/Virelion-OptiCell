# Changelog

All notable OptiCell changes are documented here.

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
