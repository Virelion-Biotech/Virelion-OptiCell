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

**Default backend is `auto`:**

| Condition | Backend used |
|-----------|----------------|
| Cellpose installed | **cellpose** (Stage-3 CTC TRA winner on 6/6 sequences) |
| Cellpose missing | **threshold** (strong BBBC039 counts / instance F1) |

Force CPU threshold anytime: `--backend threshold`.

Collapse guard: if frame-to-frame `count_cv > 1.0`, the workflow prints a warning (threshold/hybrid FP explosion pattern seen on SIM+02).

Multi-backend compare + selection v2:

```bash
python scripts/run_stage3_orchestrate.py /path/to/frames -o outputs/s3 \
  --backends threshold,hybrid,cellpose --gpu --enable-tracking
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

### Stage 3 — CTC TRA (threshold vs Cellpose)

Cellpose TRA best on **all six** sequences (e.g. GOWT1 01: 0.97 vs threshold 0.36).  
Full tables: [`outputs/stage3/STAGE3_REPORT.md`](outputs/stage3/STAGE3_REPORT.md).

---

## Install

```bash
pip install -e .                 # threshold / hybrid core
pip install -e '.[cellpose]'     # enables default auto → cellpose
```

## Other commands

```bash
opticell /path/to/images -o qc_summary.csv
python scripts/run_bbbc039_validation.py --backend cellpose --gpu --max-images 200
python scripts/validate_stage3_selection.py   # offline 6/6 selection check
```

## Roadmap

[`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md) — Stages 1–3 measured; UI / ugly-data next.

## License

AGPL-3.0-or-later. See `LICENSE`.
