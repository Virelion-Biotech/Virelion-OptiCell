# OptiCell validation — 7 October 2026

**OptiCell is not fully scientifically validated.** Reproducible software defects were repaired, but substantial segmentation, tracking, counting and QC accuracy limitations remain.

This new run covers 52,670 dataset records across ten dataset families (six CTC datasets counted as one family). These are not independent biological replicates. LIVECell contains 2,134 official annotation records but 2,081 distinct source filenames; BBBC006 contains repeated depths at 768 sites on one plate. Backends and repeated reruns are not added to the coverage total.

## Complete segmentation runs

| Dataset/backend | Records | Foreground Dice | Instance IoU F1 | Count MAE |
|---|---:|---:|---:|---:|
| BBBC039 threshold | 200 | 0.9249 | 0.7663 | 28.89 |
| BBBC039 adaptive | 200 | 0.9278 | 0.8736 | 24.34 |
| BBBC038 threshold | 670 | 0.8556 | 0.6940 | 14.06 |
| BBBC038 adaptive | 670 | 0.6090 | 0.5130 | 25.75 |
| LIVECell val threshold | 570 | 0.1131 | 0.0037 | 150.52 |
| LIVECell test threshold | 1564 | 0.0954 | 0.0038 | 153.10 |

Instance IoU F1 uses one-to-one matches at IoU ≥0.5. It is distinct from the older 20-pixel centroid F1. BBBC039 uses source-author-equivalent equal-valued red-channel decoding with 8-connectivity, retaining all three empty references. LIVECell uses official pycocotools rasterization, preserves every official ID, and projects overlapping annotations first-instance-wins; this is not official COCO AP. BBBC038 uses all 670 labeled stage1_train records.

Seeded 2,000-resample intervals are supplied in MEASURED_SUMMARY.json. BBBC039 resamples source plates; LIVECell resamples cell line/acquisition wells; BBBC038 resamples fields because biological grouping is unavailable. These intervals do not quantify donor-level or clinical uncertainty.

Official BBBC039 training/validation/test partitions are retained in the summary. They are already exposed public data, not pristine prospective holdouts.

## Counting and blur stress tests

| Dataset | Evidence | Records | Count MAE | Foreground Dice |
|---|---|---:|---:|---:|
| BBBC001 | measured | 6 | 70.25 | Unavailable |
| BBBC004 | synthetic | 100 | 92.61 | 0.9398 |
| BBBC005 | synthetic | 19200 | 26.73 | 0.8344 |

BBBC001 uses the mean of two human counters for six HT29 fields: mean relative error 18.80%. BBBC004 contains 300 simulated objects per field across five overlap levels. BBBC005 includes both stain channels, all blur levels and all 1,200 available binary references. Its fully-in-focus F1-versus-other-blur AUROC is 0.9887. Binary references cannot establish instance separation, and synthetic success does not establish biological accuracy.

## CellFMCount

All 3,023 source images and dot references were checked against the supplied metadata; every dot count agrees with its metadata count. Fixed threshold count MAE is 244.25 over the full archive and 248.37 across the 605 metadata test records. The paper evaluates DAPI images; the matching 275 DAPI test records have MAE 204.15. Keep the non-DAPI test channels separate. The source paper uses trained models and resized images, so this is an untrained full-resolution baseline characterization rather than a controlled model comparison.

All 518 empty references remain included in absolute-error metrics. Relative error is null on empty references. Four images have one out-of-bounds dot each: IDs 1285, 1401, 1186 (trainval), and 1416 (test). The author counts are preserved and the coordinate issue is recorded. No dots were moved or deleted. Coordinate-localization scoring would require clarification.

## Focus and artifact QC

BBBC006 covers all 34 official planes: 26,112 nuclear-channel images, 768 sites, 384 wells. The shipped normalized-image focus score has expert-label AUROC 0.3458, sensitivity 0.4124 and specificity 0.3819 at its fixed threshold of 100. This reproduces the existing poor result. Native-intensity Laplacian AUROC is 0.7272 as an exploratory control; it was not substituted into production defaults.

AutoQC-Bench covers the complete 148-frame test pool (100 normal, 48 anomalous): fixed production flags have sensitivity 0.6667, specificity 0.3700, accuracy 0.4662. Low-focus anomaly AUROC is 0.4650. A generic focus threshold is not a validated artifact classifier. Artifact masks are not cell masks. The five published fold partitions reuse this pool and are not independent studies.

## Full-sequence tracking

| Dataset | Sequence | Kind | Frames | TRA | SEG-style IoU |
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
| Fluo-C2DL-MSC | 01 | measured | 48 | 0.3226 | 0.1814 |
| Fluo-C2DL-MSC | 02 | measured | 48 | 0.0713 | 0.2460 |

All 12 sequences and 1,077 frames executed successfully. Tracking uses independent traccuracy 0.4.3 CTC matching. Sparse gold SEG is scored with an independent Python SEG-style implementation (>50% GT coverage), not the official CTC executable. Distances are in pixels; linking limit is 50 pixels and maximum gap is 1. Challenge training sequences were used; no challenge submission or hidden-test claim is made. Workflow completion and heuristic confidence do not imply accurate tracking.

## Learned-model operational panel

A fresh GitHub checkout executed 11 fixed panel images: the first/middle/last BBBC039 fields and the lowest official LIVECell validation ID per cell type. This is an operational smoke panel, not a representative full-corpus validation or an untouched holdout.

| Panel | Backend | Images | Dice | Instance IoU F1 | Count MAE |
|---|---|---:|---:|---:|---:|
| LIVECell_val | cellpose | 8 | 0.9076 | 0.8777 | 13.62 |
| LIVECell_val | hybrid | 8 | 0.8227 | 0.7644 | 72.25 |
| LIVECell_val | auto | 8 | 0.9076 | 0.8777 | 13.62 |
| BBBC039 | cellpose | 3 | 0.9715 | 0.9631 | 4.33 |
| BBBC039 | hybrid | 3 | 0.9541 | 0.8611 | 14.33 |
| BBBC039 | auto | 3 | 0.9715 | 0.9631 | 4.33 |

Cellpose and hybrid share a single inference per image. Auto resolves to Cellpose and reuses the identical result; it is not independent evidence. The hybrid switch performs worse on this panel. Cellpose 4.2.1.1, requested checkpoint cpsam, SHA-256 `e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2`. CPU inference only. Every panel checkpoint binds code, package, decoded-reference, image and model hashes.

The attempted 200-image Cellpose CPU run was interrupted after 19 completed fields during the twentieth inference, after about 13 minutes. Its log is retained as an incomplete attempt and its partial observations are excluded from all complete-corpus endpoint tables. The first partially completed learned panel is also retained separately and excluded. A complete GPU/Cellpose/hybrid matrix is still uncompleted.

## Repairs, reruns and reproducibility

The code fixes reject ambiguous BBBC039 source/reference names and nested roots, multiple BBBC038 source images, unknown Cellpose checkpoints, silent default-model retries and silent GPU-to-CPU substitution. Cellpose 3 named-model selection uses model_type; Cellpose 4 uses pretrained_model. Regression tests cover these failure modes.

**216 tests pass; configured lint passes.** A fresh GitHub checkout of commit `5a8990441e448d5f1c5a99f64a08233f03734956` passed its then-current 215 tests and reproduced every BBBC039 threshold row and every aggregate exactly. It also reproduced all AutoQC endpoint values and executed the learned panel. The subsequent Cellpose 3 argument-selection repair is covered by the additional test; it does not change the executed Cellpose 4 inference parameters.

The focus study records core SHA-256 corresponding to original main commit `e4a3590c845d52f494d3fe58799b8935e9d56c3f`. Focus, grayscale and threshold algorithms were not tuned or replaced during this run. Source snapshots, source hashes, logs, dependencies and per-image measurements are included. Raw datasets and model weights are not redistributed.

Run the shipped scripts with the official datasets. The streamed BBBC006 runner downloads one archive at a time. For a GPU extension, run `scripts/run_learned_panel.py --all --gpu` on the complete BBBC039 and LIVECell validation corpora; the existing native runners additionally support BBBC038 and LIVECell test. A requested GPU now fails explicitly if it resolves to CPU.

## Uncompleted or unsuitable data

| Resource | Disposition |
|---|---|
| TissueNet | Registration/access not available in this run; not evaluated. |
| EVICAN | Official API and annotation endpoint returned HTTP 502; not evaluated. Partially annotated fields require sparse-reference-aware scoring. |
| BBBC022/036/047, JUMP-CP, IDR, HPA | Collections, not one defined validation endpoint. Perturbation/phenotype labels cannot substitute for dense cell masks or expert image-QC labels. No numerical claim made. |
| ATOM single-cell crops | Classification data; do not establish full-field segmentation/counting accuracy. |
| Cardiac microscopy | No cardiac-specific expert reference cohort acquired; no cardiomyocyte whole-cell, maturity, sarcomere, contractility or regeneration claim validated. |
| 3D, division/lineage events, calibration, independent labs | Software tests alone do not establish biological accuracy; dedicated datasets and protocols remain necessary. |
| CellProfiler comparator | Separate comparator not executed in this run. |

The remaining low accuracy is not repaired by relabeling references, changing test labels or tuning defaults against inspected test outcomes. Next evidence needs modality-specific calibration on separate acquisition units, complete learned-backend matrices, and cardiac-specific annotated data.

## Primary sources

- [BBBC001](https://bbbc.broadinstitute.org/BBBC001), [BBBC004](https://bbbc.broadinstitute.org/BBBC004), [BBBC005](https://bbbc.broadinstitute.org/BBBC005), [BBBC006](https://bbbc.broadinstitute.org/BBBC006), [BBBC038](https://bbbc.broadinstitute.org/BBBC038), [BBBC039](https://bbbc.broadinstitute.org/BBBC039).
- [BBBC039 author decoder](https://gist.github.com/jccaicedo/15e811722fca51e3ae90e8b43057f075).
- [LIVECell](https://github.com/sartorius-research/LIVECell).
- [Cell Tracking Challenge](https://celltrackingchallenge.net/2d-datasets/).
- [CellFMCount archive](https://doi.org/10.5281/zenodo.17088532), [paper and DAPI evaluation scope](https://arxiv.org/html/2511.19351v1).
- [AutoQC-Bench archive](https://doi.org/10.6019/S-BIAD2133), [paper](https://doi.org/10.1038/s44303-025-00117-8).

Dataset licenses remain source-specific. In particular LIVECell and BBBC001/004 include non-commercial terms; BBBC005/006/038/039 are CC0; CellFMCount is CC BY-SA 4.0. This package contains analysis outputs, not redistributed raw images.
