# Virelion-OptiCell

**Headless Python toolkit** for microscopy QC, segmentation, tracking, phenotyping, and experiment-level quantitative analysis.

**Product thesis:** trustworthy measurements with explicit QC — not another black-box segmenter.

**Measured numbers only.** No fabricated benchmarks.

---

## Measured baseline — BBBC039 (U2OS Hoechst nuclei)

Public: [BBBC039](https://bbbc.broadinstitute.org/BBBC039). Same FOVs + OptiCell metrics for every row.

### Full set (n=197 nonzero-GT) — Stage 1 complete

| Backend | Dice | Instance F1 | Mean \|count err\| | Rel count |
|---------|-----:|------------:|------------------:|----------:|
| **OptiCell threshold** | 0.925 | **0.929** | **8.7** | **0.077** |
| **OptiCell hybrid** | 0.929 | 0.924 | 9.0 | 0.083 |
| **Cellpose-SAM** | **0.969** | 0.907 | 16.7 | 0.172 |
| **CellProfiler 4.2** | 0.895 | 0.722 | 28.1 | 0.281 |

Full report: [`outputs/bbbc039_validation/BBBC039_STAGE1_COMPLETE.md`](outputs/bbbc039_validation/BBBC039_STAGE1_COMPLETE.md).

**Takeaway:** OptiCell wins **counts / instance F1**; Cellpose wins **pixel Dice** on this fluorescent nuclei set.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e '.[cellpose]'   # optional
```

## Usage

```bash
opticell /path/to/images -o qc_summary.csv
python scripts/run_killer_workflow.py /path/to/images -o outputs/workflow_run --backend threshold
python scripts/run_bbbc039_validation.py --backend cellpose --gpu --max-images 200
```

## Roadmap

[`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md)

## License

AGPL-3.0-or-later. See `LICENSE`.
