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
pip install -e '.[cellpose]'   # Cellpose = default when installed

# Time-lapse folder (frames sorted by filename)
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

Optional GPU backends: `pip install -e '.[cellpose]'` then enable GPU in the sidebar.

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

| Backend | n | Dice | Instance F1 | Mean \|count err\| | Rel. count err. |
|---------|---|------|--------------|---------------------|------------------|
| **Threshold (Otsu)** | 200 | 0.846 | 0.807 | 14.1 | 0.231 |

```bash
python scripts/run_bbbc038_validation.py --max-images 200 --backend threshold
```

### LIVECell (dense phase-contrast) — adapter ready

```bash
python scripts/run_livecell_validation.py --max-images 20 --backend threshold
```

CC BY-NC 4.0 (non-commercial). ~1.3GB download on first run. **No measured README table until you run it and commit the JSON.**

---

## Install

```bash
pip install -e .                 # core
pip install -e '.[cellpose]'     # auto → cellpose
pip install -e '.[ui]'           # Streamlit app
```

## Roadmap

[`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md)

## License

AGPL-3.0-or-later. See `LICENSE`.
