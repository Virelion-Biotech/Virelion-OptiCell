# OptiCell v2.18.0 release notes

Release date: 2026-09-19

## Highlights

- Production-path segmentation orchestration uses the phase/low-contrast-aware backend suggestion logic.
- Streamlit UI smoke coverage uses the real Streamlit AppTest harness.
- The CLI killer workflow exposes an optional human-in-the-loop review queue for flagged or non-PASS FOVs, with auditable accept/reject/rerun decisions.
- Packaging now uses a single source of truth for the package version and an SPDX license expression.
- End-to-end workflow regression tests cover output artifacts and failure handling.
- LIVECell validation benchmarks the current `hybrid` and `auto` implementations on the same 20 validation FOVs.

## Scientific interpretation

LIVECell remains a difficult dense phase-contrast benchmark. OptiCell does not claim hybrid superiority from the pre-rule 0.622 Dice result. The final `auto` path is intended to route low-contrast/phase-like images to Cellpose when available; the release benchmark records the measured behavior of the current implementation.

## Scope

This release is a hardening/validation release, not a claim of clinical validation or universal microscopy segmentation performance.
