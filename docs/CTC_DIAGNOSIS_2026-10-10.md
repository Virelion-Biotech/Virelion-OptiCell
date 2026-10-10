# CTC failure diagnosis and opt-in fluorescence repair

The returned `04_OptiCell_CTC_v1_results.zip` completed without worker errors, but both classical backends performed poorly. Its 52 artifact hashes were checked. Original results are retained as baselines, not rewritten as successful scientific validation.

## Confirmed causes

- GOWT1 nuclei are approximately 50–100 pixels across, have dark internal regions, and vary substantially in brightness. Global Otsu favours bright objects, misses dim nuclei and leaves holes.
- The legacy adaptive threshold uses a 31-pixel window and requires intensity above the local mean by 3. The window fits inside large nuclei. It selects texture/rims rather than whole nuclei, causing fragmentation, insufficient reference coverage and spurious tracks.
- `confidence_score=100` measures absence of a few QC red flags. Morphology-gate `PASS` is not reference-mask accuracy. Outputs now explicitly identify uncalibrated QC, unassessed accuracy and morphology-only acceptance; PASS reasons also state that accuracy is unassessed.
- Tracking links centroids one-to-one. The CTC exporter writes parent ID zero for all tracklets, so it does not reconstruct mitosis. Reference files contain 5 nonzero-parent daughter records in sequence 01 and 13 in sequence 02. Division-aware linking remains unresolved. Fragmentation and missed objects also damage tracking.

## Repair and reproduction

The explicit `fluorescence` backend smooths the image, applies Otsu to log-compressed intensities, fills enclosed foreground holes, and applies distance watershed with an image-derived neighbourhood. Area admission is applied again after splitting. Uniform images produce no objects. No weights or GPU are needed.

Existing threshold/adaptive, hybrid selection and auto defaults are unchanged, preserving previous BBBC039 measurements. Request the new backend explicitly:

```bash
pip install -e '.[validation,fluorescence]'
python scripts/run_ctc_cloud_validation.py \
  --data-dir /path/to/extracted-data --out-dir outputs/ctc_fluorescence \
  --datasets Fluo-N2DH-GOWT1 --backend fluorescence
```

It is also available through `opticell.segmentation.get_backend('fluorescence')` and `scripts/run_killer_workflow.py --backend fluorescence`. It assumes bright fluorescent objects on darker background; phase contrast, inverted staining and other modalities require separate validation.

## Full-sequence characterization

Both sequences have 92 raw frames. SEG uses sparse gold annotations: 30 frames/144 objects in 01 and 20 frames/121 objects in 02. Scores use independent Python SEG-style and traccuracy CTCMetrics, not official challenge executables. Higher is better.

| Backend | Sequence | SEG-style IoU | DET | TRA | LNK |
|---|---|---:|---:|---:|---:|
| Threshold baseline | 01 | 0.2127 | 0.3670 | 0.3768 | 0.4428 |
| Adaptive baseline | 01 | 0.0586 | 0.0000 | 0.0000 | 0.0192 |
| Fluorescence repair | 01 | 0.5009 | 0.7815 | 0.7805 | 0.7731 |
| Threshold baseline | 02 | 0.4306 | 0.3089 | 0.3179 | 0.3794 |
| Adaptive baseline | 02 | 0.0042 | 0.0000 | 0.0000 | 0.0012 |
| Fluorescence repair | 02 | 0.7280 | 0.6813 | 0.6853 | 0.7122 |

All 184 frames completed. Original and newly downloaded data archive SHA256 match: `1a7bd9a7d1d10c4122c7782427b437246fb69cc3322a975485c04e206f64fc2c`.

The repair was designed after inspecting this public training dataset and baseline errors. Sequence 01 was the initial development probe; the same numerical implementation was evaluated on both full sequences without a parameter sweep. Both sequences had been inspected during diagnosis. **These are development characterization results, not a blinded holdout or qualification.** Very dim nuclei, false positives and incomplete lineage reconstruction remain. No acceptance cutoff was predeclared. No biological, clinical or general modality validation is claimed.

Full per-frame SEG scores, tracking errors, baseline summaries, environment and input/source/prediction hashes are in [`outputs/ctc_diagnosis_2026_10_10`](../outputs/ctc_diagnosis_2026_10_10). Large raw data and masks are not committed; the command reproduces them from the named archive.

## Verification

Regression cases cover dim/bright nuclei wider than the adaptive window, dark interiors, touching nuclei, uniform fields, area rejection and invalid dtype. Existing reference-decoding and majority-overlap tests remain. Full test and CI results are recorded in the pull request.

References: [CTC datasets](https://celltrackingchallenge.net/2d-datasets/), [evaluation methodology](https://celltrackingchallenge.net/evaluation-methodology/).
