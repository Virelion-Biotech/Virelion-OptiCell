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

**Default backend is `auto`:** Cellpose if installed, else threshold.

---

## Easy UI (Stage 4)

```bash
pip install -e '.[ui]'
streamlit run app_streamlit.py
```

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

Public: [LIVECell](https://github.com/sartorius-research/LIVECell) · CC BY-NC 4.0 · mean **197.6** GT cells/image (max 425).

| Backend | n | Dice | Instance F1 | Mean \|count err\| |
|---------|---|------|--------------|------------------:|
| **Cellpose-SAM (GPU)** | 20 | **0.930** | **0.904** | **25.7** |
| Hybrid (collapse-aware) | 20 | 0.622 | 0.767 | 37.1 |
| Threshold (Otsu) | 20 | 0.054 | 0.435 | 87.5 |

- Collapse fix routes true zero-threshold FOVs to Cellpose (5/20).
- Hybrid still loses when threshold finds sparse false blobs (7/20) — pure Cellpose is the right default here.
- Reports: [`LIVECELL_CELLPOSE_N20_REPORT.md`](outputs/livecell_validation/LIVECELL_CELLPOSE_N20_REPORT.md), [`LIVECELL_THRESHOLD_N20_REPORT.md`](outputs/livecell_validation/LIVECELL_THRESHOLD_N20_REPORT.md), [`LIVECELL_HYBRID_N20_REPORT.md`](outputs/livecell_validation/LIVECELL_HYBRID_N20_REPORT.md).

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
