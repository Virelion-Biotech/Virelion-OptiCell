# OptiCell 2.19.0 — Research Platform

OptiCell 2.19.0 turns the existing analysis engine and single-FOV workbench into
a persistent research platform.

## Product workflows

- persistent projects, samples, groups, conditions, replicates and saved runs;
- multi-image batch analysis with dataset-level QC and downloadable run bundles;
- auditable human review decisions with corrected-mask attachments;
- model presets, custom Cellpose checkpoints and configuration fingerprints;
- empirical QC calibration tools and a validation-evidence registry;
- OME-TIFF large-data planning/streaming and pyramid inspection;
- optional local authentication, quotas, audit logs, health checks and backups;
- Docker/Compose reference deployment and tag-based release automation.

## Reproducibility

Manifest v2 records input SHA-256 hashes, model hashes when available, source
commit, package versions, runtime identifiers, parameters and a digest of the
manifest itself.

## Validation boundary

This release does **not** claim universal scientific calibration or independent
external-laboratory validation. The calibration UI needs labeled outcomes, and
the evidence registry currently marks the repository's BBBC039, CTC TRA and
LIVECell studies as non-independent evidence. See
`docs/SCIENTIFIC_VALIDATION.md` and `docs/PRODUCT_COMPLETENESS.md`.
