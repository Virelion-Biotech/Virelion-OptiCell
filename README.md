# Virelion-OptiCell

**Headless Python toolkit** for microscopy QC, segmentation, tracking, phenotyping, and experiment-level quantitative analysis.

**Product thesis:** trustworthy measurements with explicit QC — not another black-box segmenter.

**Measured numbers only.** No fabricated benchmarks.

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

## Easy UI (Stage 4)

```bash
pip install -e '.[ui,cellpose]'
streamlit run app_streamlit.py
```

Upload one image → acquisition QC → segmentation → acceptance gate → per-object CSV + overlay download.

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

### LIVECell (dense phase-contrast, val, same 20 FOVs)

| Backend | n | Dice | Instance F1 | Mean \|count err\| |
|---------|---|------|--------------|------------------:|
| **Cellpose-SAM (GPU)** | 20 | **0.930** | **0.904** | **25.7** |
| Hybrid | 20 | 0.622 | 0.767 | 37.1 |
| Threshold (Otsu) | 20 | 0.054 | 0.435 | 87.5 |

---

## Install

```bash
pip install -e .
pip install -e '.[cellpose]'
pip install -e '.[ui]'
```

## Roadmap

[`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md)

## License

AGPL-3.0-or-later. See `LICENSE`.
