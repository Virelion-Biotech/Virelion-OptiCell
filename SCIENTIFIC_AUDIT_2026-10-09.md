# Scientific audit changes — 2026-10-09

## Behavior

Disable silent Cellpose-to-threshold fallback by default. Exploratory fallback requires allow_fallback=True and retains an error flag. Add fail-closed scientific-domain report checks tied to the executing threshold-code fingerprint.

## Scope and remaining evidence

Cellpose scientific mode is blocked because binding executing weight identity is not implemented. A declared report can be checked for integrity/domain but is not independently authenticated or scientifically assessed here. No modality backend, segmentation confidence map or tracking benchmark has been trained/qualified.

## Implementation

- `opticell/domain_gate.py`
- `tests/test_scientific_domain_gate.py`
- `opticell/hearttwin_adapter.py`
- `qc_pipeline.py`

## Verification

Regression tests accompany the changes. Repository test results are recorded in the audit completion report and draft pull request. Software regression checks do not establish numerical, biological, transport or clinical validity.
