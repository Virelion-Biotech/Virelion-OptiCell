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

## Measured baselines

### Stage 1 — BBBC039 nuclei (n=197)

| Backend | Dice | Instance F1 | Mean \|count err\| |
|---------|-----:|------------:|------------------:|
| OptiCell threshold | 0.925 | **0.929** | **8.7** |
| OptiCell hybrid | 0.929 | 0.924 | 9.0 |
| Cellpose-SAM | **0.969** | 0.907 | 16.7 |
| CellProfiler 4.2 | 0.895 | 0.722 | 28.1 |

### Stage 3 — CTC TRA

Cellpose TRA best on **all six** sequences. See [`outputs/stage3/STAGE3_REPORT.md`](outputs/stage3/STAGE3_REPORT.md).

### BBBC038 (Kaggle 2018 DSB nuclei)

| Backend | n | Dice | Instance F1 | Mean \|count err\| |
|---------|---|------|--------------|------------------:|
| Threshold (Otsu) | 200 | 0.846 | 0.807 | 14.1 |

### LIVECell (dense phase-contrast, validation, same 20 FOVs)

| Backend | n | Dice | Instance F1 | Mean \|count err\| |
|---------|---|------|--------------|------------------:|
| Cellpose-SAM (GPU) | 20 | **0.930** | **0.904** | **25.7** |
| Hybrid (GPU) | 20 | **0.930** | **0.904** | **25.7** |
| Auto (GPU) | 20 | **0.930** | **0.904** | **25.7** |
| Threshold (CPU) | 20 | 0.054 | 0.435 | 87.5 |

The current hybrid/auto measurements supersede the historical pre-routing-rule hybrid result; see [the current n=20 report](outputs/livecell_validation/LIVECELL_CURRENT_N20_REPORT.md).

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
