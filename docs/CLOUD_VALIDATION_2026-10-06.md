# Cloud validation, 6 October 2026

**The software suite passes; broad scientific validation does not.** All runs used cloud CPUs (9 cores, 8 GiB), with no personal computer or self-hosted GPU workflow. CardiEval was excluded.

## Complete CPU studies

| Data / backend | Records | Dice | Instance IoU F1 | Count MAE |
|---|---:|---:|---:|---:|
| BBBC039 threshold | 200 | 0.9249 | 0.7663 | 28.89 |
| BBBC039 adaptive fixed | 200 | 0.9278 | 0.8736 | 24.34 |
| BBBC038 threshold | 670 | 0.8556 | 0.6940 | 14.06 |
| BBBC038 adaptive fixed | 670 | 0.6090 | 0.5130 | 25.75 |
| LIVECell validation threshold | 570 | 0.1131 | 0.0037 | 150.52 |
| LIVECell test threshold | 1564 | 0.0954 | 0.0038 | 153.10 |

IoU F1 uses one-to-one matches at IoU ≥0.5. It differs from the previous 20-pixel centroid F1. Old BBBC039 reference decoding merged touching nuclei, so GT-dependent legacy instance/count comparisons are withdrawn. Foreground Dice alone does not demonstrate instance separation.

All 200 BBBC039 records include the three empty references. BBBC038 uses all 670 labeled stage1_train records. LIVECell preserves all official 570 validation and 1564 test COCO IDs; these correspond to 569 and 1512 distinct source files. Repeated filenames are not independent images. Official COCO polygons are decoded by pycocotools and projected first-instance-wins at overlaps; these values are not official COCO AP.

Correcting the adaptive threshold offset stopped selecting dark halos around bright cells. BBBC039 IoU F1 rose from 0.0000 to 0.8736; BBBC038 rose from 0.0576 to 0.5130. Threshold still performs better on heterogeneous BBBC038. No universal backend rule is inferred from the improvement.

## Tracking and focus

| CTC dataset | Sequence | Kind | Frames | TRA | SEG-style IoU |
|---|---|---|---:|---:|---:|
| Fluo-N2DH-GOWT1 | 01 | measured | 92 | 0.3768 | 0.2127 |
| Fluo-N2DH-GOWT1 | 02 | measured | 92 | 0.3179 | 0.4306 |
| Fluo-N2DH-SIM+ | 01 | simulated | 65 | 0.8275 | 0.6180 |
| Fluo-N2DH-SIM+ | 02 | simulated | 150 | 0.0000 | 0.1053 |
| Fluo-N2DL-HeLa | 01 | measured | 92 | 0.6697 | 0.4516 |
| Fluo-N2DL-HeLa | 02 | measured | 92 | 0.7318 | 0.5176 |
| DIC-C2DH-HeLa | 01 | measured | 84 | 0.0000 | 0.0000 |
| DIC-C2DH-HeLa | 02 | measured | 84 | 0.0000 | 0.0000 |
| PhC-C2DH-U373 | 01 | measured | 115 | 0.0000 | 0.0140 |
| PhC-C2DH-U373 | 02 | measured | 115 | 0.0000 | 0.0329 |

Ten full CTC sequences cover 981 frames: 766 measured and 215 simulated. TRA/DET/LNK use independent traccuracy 0.4.3 CTC matching; sparse gold SEG uses an independent Python SEG-style scorer (GT overlap >50%), not the official challenge executable. Linking distance is 50 pixels and maximum gap is 1. These are public labeled training sequences, not independent challenge test submissions.

BBBC006 nuclear w1 (Hoechst) covers **all 34 published planes, 26,112 images, 768 sites and 384 wells**. The shipped normalized-working-image focus score has expert-label AUROC **0.3458**, sensitivity 0.4124 and specificity 0.3819 at its fixed threshold of 100. Site grouping now preserves physical wells/sites rather than image UUIDs. Alternative native-intensity controls are exploratory and were not silently substituted into production defaults.

On LIVECell, every threshold record with heuristic confidence 100 had IoU F1 below 0.8 (432 validation and 1135 test records). This is a descriptive failure endpoint, not a calibrated automatic-acceptance rule.

## Uncompleted scope

Cellpose-SAM CPU smoke tests cover five nuclei images (Dice 0.9655, IoU F1 0.9326) and one LIVECell auto case, resolved to Cellpose (Dice 0.9342, IoU F1 0.9420). These prefix cases are not representative full-data studies. Full Cellpose/hybrid/auto matrices and independent laboratory qualification are uncompleted. Public training data and test outcomes were inspected during repairs; no pristine held-out or unknown-pretraining-exposure claim is made.

## Repairs and checks

Author-equivalent BBBC039 instance decoding, official COCO rasterization, shape metrics, fail-closed reference and dimension admission, proper CSV quoting, scalar streaming aggregation, physical z-stack grouping, tied-rank handling, bounded-memory label statistics, correct adaptive polarity, exact COCO-ID coverage, and source/code/package/image-verified checkpoint reuse are implemented. Missing references or ambiguous source files cannot silently shrink a study.

**204 tests pass.** Configured lint passes. GitHub CI passes Python 3.10/3.11/3.12, security/dependency checks, isolated wheel/sdist installation and container smoke. Cold GitHub-checkout reruns reproduced all BBBC039 threshold measurements and both GOWT1 sequences. Vectorized calculations preserved every measured value on the 200-image and 670-image threshold runs. Interrupted memory/storage attempts were excluded; final counts require complete result files and exact official-ID coverage.

## Reproduce

Install `pip install -e ".[dev,validation]"`, retrieve the official datasets, and run:

```bash
python scripts/run_bbbc039_validation.py --data-dir data/bbbc039 --backend threshold --max-images 0 --include-empty-gt --skip-download
python scripts/run_bbbc039_validation.py --data-dir data/bbbc039 --backend adaptive --max-images 0 --include-empty-gt --skip-download
python scripts/run_bbbc038_validation.py --data-dir data/bbbc038 --backend threshold --max-images 0 --include-empty-gt --skip-download
python scripts/run_bbbc038_validation.py --data-dir data/bbbc038 --backend adaptive --max-images 0 --include-empty-gt --skip-download
python scripts/run_livecell_complete.py --data-dir data/livecell --out-dir outputs/livecell_complete --workers 1
python scripts/run_ctc_cloud_validation.py --data-dir data/ctc --out-dir outputs/ctc_complete --datasets Fluo-N2DH-GOWT1 Fluo-N2DH-SIM+ Fluo-N2DL-HeLa DIC-C2DH-HeLa PhC-C2DH-U373
python scripts/run_bbbc006_focus_qc.py --root data/bbbc006 --max-sites 0 --expected-z {0..33}
pytest
```

For LIVECell, put official annotations in `data/livecell/raw` and extracted images under `data/livecell/images`. For BBBC006, extract nuclear w1 TIFFs under `z_00` … `z_33`; the focus runner does not download them automatically. This study streamed archives to avoid retaining the full raw series.

Aggregate measured evidence: [MEASURED_SUMMARY.json](../outputs/cloud_validation_2026_10_06/MEASURED_SUMMARY.json). The detailed owner report contains per-image CSV/JSON, cluster intervals, source hashes, model/environment versions and full focus/tracking scores. Raw datasets and model weights are not redistributed.

Primary sources: [BBBC039](https://bbbc.broadinstitute.org/BBBC039), [author decoder](https://gist.github.com/jccaicedo/15e811722fca51e3ae90e8b43057f075), [BBBC038](https://bbbc.broadinstitute.org/BBBC038), [LIVECell](https://github.com/sartorius-research/LIVECell), [CTC](https://celltrackingchallenge.net/2d-datasets/), [BBBC006](https://bbbc.broadinstitute.org/BBBC006).
