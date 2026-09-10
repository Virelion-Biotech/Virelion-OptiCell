# Virelion-OptiCell

**Headless Python toolkit** for microscopy QC, segmentation, tracking, phenotyping, and experiment-level quantitative analysis.

**Product thesis:** given raw microscopy data, OptiCell should produce *trustworthy* measurements with less manual work and explicit QC — not another black-box segmenter.

We publish **measured** numbers only. No fabricated benchmarks.

---

## Measured baseline — BBBC039 (U2OS Hoechst nuclei)

Public dataset: [BBBC039](https://bbbc.broadinstitute.org/BBBC039). Same FOV list + OptiCell `validation.py` for every row.

### Full set (n≈197–200 nonzero-GT)

| Backend | Dice | Instance F1 | Mean \|count err\| | Rel count |
|---------|-----:|------------:|------------------:|----------:|
| **OptiCell threshold** | 0.925 | **0.929** | **8.7** | **0.077** |
| **OptiCell hybrid** | **0.929** | 0.924 | 9.0 | 0.083 |
| **CellProfiler 4.2** | 0.895 | 0.722 | 28.1 | 0.281 |

### Pilot n=50 (includes Cellpose)

| Backend | Dice | Instance F1 | Mean \|count err\| |
|---------|-----:|------------:|------------------:|
| OptiCell threshold | 0.945 | **0.955** | **6.1** |
| OptiCell hybrid | 0.951 | 0.950 | 6.4 |
| Cellpose-SAM | **0.970** | 0.908 | 15.0 |
| CellProfiler | 0.897 | 0.722 | 26.2 |

Reports: `outputs/bbbc039_validation/BBBC039_FULLSET_COMPARISON.md`, `BBBC039_N50_FULL_COMPARISON.md`.

---

## What it contains

- Acquisition QC · threshold / adaptive / Cellpose / hybrid segmentation
- Validation: IoU, Dice, instance F1, count error
- Tracking, features, plate QC, provenance

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e '.[cellpose]'   # optional
```

## Usage

```bash
opticell /path/to/images -o qc_summary.csv
python scripts/run_killer_workflow.py /path/to/images -o outputs/workflow_run --backend threshold
python scripts/run_bbbc039_multi_backend.py --max-images 200 --backends threshold,hybrid --skip-download
```

## Roadmap

[`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md)

## License

AGPL-3.0-or-later. See `LICENSE`.
