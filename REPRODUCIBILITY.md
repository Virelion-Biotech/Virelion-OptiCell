# Reproducibility checklist

Use this checklist when turning an OptiCell analysis into a report, figure, or publication.

## Acquisition

- [ ] Record microscope/modality, magnification, objective, exposure, and relevant acquisition settings.
- [ ] Preserve raw data separately from derived outputs.
- [ ] Record pixel/voxel size and time interval when applicable.
- [ ] Inspect acquisition-artifact and plate-QC outputs before downstream analysis.

## Experimental design

- [ ] Define the biological experimental unit.
- [ ] Distinguish biological from technical replicates.
- [ ] Predefine controls, blocking, and exclusion criteria.
- [ ] Avoid cell-level pseudoreplication in inferential statistics.

## Analysis

- [ ] Record OptiCell version/commit.
- [ ] Preserve analysis parameters and input SHA-256 hashes.
- [ ] Preserve the deterministic analysis fingerprint.
- [ ] Record segmentation/tracking backend and model version when applicable.
- [ ] Run sensitivity/robustness analysis for important tunable parameters.
- [ ] Benchmark segmentation/tracking against representative ground truth when used as a primary endpoint.

## Outputs

- [ ] Preserve machine-readable result tables.
- [ ] Preserve provenance/manifests and experiment-audit output.
- [ ] Preserve QC decisions and reasons, not only a composite score.
- [ ] Record any manual exclusions or corrections.
- [ ] Archive the exact environment or built artifact used for the analysis when feasible.

## Interpretation

OptiCell QC and robustness scores are evidence for evaluating an analysis pipeline, not guarantees of biological correctness. Thresholds are project-specific unless independently validated.
