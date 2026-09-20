# OptiCell v2.18.0 release notes

Release date: 2026-09-19

## Highlights

- Production-path segmentation orchestration uses the phase/low-contrast-aware backend suggestion logic.
- Streamlit UI smoke coverage uses the real Streamlit AppTest harness.
- The CLI killer workflow exposes an optional human-in-the-loop review queue for flagged or non-PASS FOVs, with auditable accept/reject/rerun decisions.
- Packaging now uses a single source of truth for the package version and an SPDX license expression.
- End-to-end workflow regression tests cover output artifacts and failure handling.
- LIVECell validation benchmarks the current `cellpose`, `hybrid`, and `auto` paths on the same 20 validation FOVs, with a threshold baseline.
- The measured T4 n=20 results are recorded in `outputs/livecell_validation/`; ordinary CI validates those committed artifacts instead of attempting GPU-style inference on CPU runners.

## Scientific interpretation

LIVECell remains a difficult dense phase-contrast benchmark. OptiCell does not claim hybrid superiority from the pre-rule 0.622 Dice result. The final `auto` path is intended to route low-contrast/phase-like images to Cellpose when available; the release benchmark records the measured behavior of the current implementation.

## Scope

This release is a hardening/validation release, not a claim of clinical validation or universal microscopy segmentation performance.


## LIVECell T4 n=20 measured results

Environment: Google Colab, NVIDIA Tesla T4, CUDA-enabled PyTorch, Cellpose-SAM `cpsam`, validation split, same 20 FOVs.

| Backend | Dice | IoU | Instance F1 | Mean absolute count error | Relative count error |
|---|---:|---:|---:|---:|---:|
| Cellpose-SAM | 0.9299 | 0.8694 | 0.9042 | 25.65 | 10.45% |
| Hybrid | 0.9299 | 0.8694 | 0.9042 | 25.65 | 10.45% |
| Auto | 0.9299 | 0.8694 | 0.9042 | 25.65 | 10.45% |
| Threshold | 0.0543 | 0.0284 | 0.4352 | 87.50 | 32.90% |

The auto run resolved all 20 sampled FOVs to Cellpose. These results are a single n=20 validation slice and are not claims of universal segmentation performance.
