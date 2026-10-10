# Virelion-OptiCell

**Headless Python toolkit** for microscopy QC, segmentation, tracking, phenotyping, and experiment-level quantitative analysis.

---

## Quickstart (recommended)

```bash
git clone https://github.com/Virelion-Biotech/Virelion-OptiCell.git
cd Virelion-OptiCell
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[cellpose]'

python scripts/run_killer_workflow.py /path/to/frames \
  -o outputs/run --enable-tracking --gpu
```

**Default backend is `auto`:** Cellpose if installed (always preferred on low-contrast / phase-like images), else threshold.

**5-minute demo:** [`docs/DEMO_5_MIN.md`](docs/DEMO_5_MIN.md)

---

## Research Workbench (Stage 4)

```bash
pip install -e '.[ui,cellpose]'
streamlit run app_streamlit.py
```

The multipage workbench now supports:

- single-image analysis with QC, acceptance and reproducibility exports,
- persistent projects, samples, conditions and replicates,
- multi-image batch analysis with dataset/outlier/replicate/plate QC,
- human accept/reject/rerun decisions and corrected-mask attachment,
- segmentation presets and custom checkpoint hashing,
- empirical QC-score calibration against labeled outcomes,
- OME-TIFF/z-stack/multichannel/time-series streaming plans,
- health/audit/backup operations.

The default local deployment uses SQLite. See [production deployment](docs/PRODUCTION_DEPLOYMENT.md) and the [product completeness matrix](docs/PRODUCT_COMPLETENESS.md).

---

## Measured validation

The [10 October GOWT1 diagnosis and repair](docs/CTC_DIAGNOSIS_2026-10-10.md) adds an explicit CPU `fluorescence` backend. Full-sequence TRA improves from 0.377/0.318 (threshold) to 0.780/0.685, with SEG-style 0.501/0.728. These are public training-data development measurements, not independent qualification; division-aware tracking and dim-object errors remain. Install `.[fluorescence]` and request `--backend fluorescence`. Existing backend defaults are preserved.

The 6 October 2026 cloud CPU study corrected reference decoding and adaptive thresholding, then scored complete public data. **The software tests pass; broad scientific validation fails.**

| Data / backend | Records | Dice | Instance IoU F1 | Count MAE |
|---|---:|---:|---:|---:|
| BBBC039 threshold | 200 | 0.9249 | 0.7663 | 28.90 |
| BBBC039 adaptive, corrected | 200 | 0.9278 | 0.8736 | 24.35 |
| BBBC038 threshold | 670 | 0.8556 | 0.6940 | 14.06 |
| BBBC038 adaptive, corrected | 670 | 0.6090 | 0.5130 | 25.75 |
| LIVECell validation, threshold | 570 | 0.1131 | 0.0037 | 150.52 |
| LIVECell test, threshold | 1564 | 0.0954 | 0.0038 | 153.10 |

IoU F1 matches instance shapes at IoU ≥0.5; the old centroid F1 was a different metric. GT-dependent legacy BBBC039 instance/count comparisons are withdrawn because touching reference nuclei were merged. LIVECell records include repeated source files and use a documented overlap projection, not official COCO AP.

The study also covers 10 tracking sequences (981 frames, including 215 simulated frames) and all 34 BBBC006 focus depths (26,112 images). Several tracking datasets fail, and the shipped focus score has AUROC 0.3458. Cellpose-SAM was smoke-tested on five nuclei images and one LIVECell auto case; its full learned-backend matrices remain uncompleted.

See [methods, repairs, scope and results](docs/CLOUD_VALIDATION_2026-10-06.md) and [the measured summary](outputs/cloud_validation_2026_10_06/MEASURED_SUMMARY.json). Historical GPU panels and older centroid-based comparisons remain in their original output folders and are not substitutes for the current corrected studies.

---

## Install

```bash
pip install -e .
pip install -e '.[cellpose]'
pip install -e '.[ui]'
```

### Container

```bash
docker compose up --build -d
```

## Validation and release

- Current LIVECell n=20 measurements: [`outputs/livecell_validation/LIVECELL_CURRENT_N20_REPORT.md`](outputs/livecell_validation/LIVECELL_CURRENT_N20_REPORT.md)
- Release notes: [`docs/RELEASE_NOTES_v2.19.0.md`](docs/RELEASE_NOTES_v2.19.0.md)
- Optional HITL review: run the killer workflow with `--write-review-queue`, then record decisions with `--review-decisions reviewed.csv`.

## Roadmap

[`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md)

## License

AGPL-3.0-or-later. See `LICENSE`.
