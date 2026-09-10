# Virelion-OptiCell

**Headless Python toolkit** for microscopy QC, segmentation, tracking, phenotyping, and experiment-level quantitative analysis.

**Product thesis:** given raw microscopy data, OptiCell should produce *trustworthy* measurements with less manual work and explicit QC — not another black-box segmenter.

We publish **measured** numbers only. No fabricated benchmarks.

---

## Measured baseline — BBBC039 (U2OS Hoechst nuclei)

Public dataset: [BBBC039](https://bbbc.broadinstitute.org/BBBC039) · same FOV list and OptiCell `validation.py` metrics for every row.

### n=50 (pilot, fully compared)

| Backend | Dice | Instance F1 | Mean \|count err\| |
|---------|-----:|------------:|------------------:|
| **OptiCell threshold** | 0.945 | **0.955** | **6.1** |
| **OptiCell hybrid** | 0.951 | 0.950 | 6.4 |
| **Cellpose-SAM** | **0.970** | 0.908 | 15.0 |
| **CellProfiler 4.2** | 0.897 | 0.722 | 26.2 |

### n=200 (OptiCell backends)

| Backend | Dice | Instance F1 | Mean \|count err\| | Rel. count (197 FOVs) |
|---------|-----:|------------:|------------------:|----------------------:|
| Threshold | 0.925 | **0.929** | **8.7** | **0.077** |
| Hybrid | **0.929** | 0.924 | 9.0 | 0.083 |

Details: `outputs/bbbc039_validation/` (including `BBBC039_N50_FULL_COMPARISON.md`).

```bash
pip install -e '.[cellpose]'   # optional
python scripts/run_bbbc039_multi_backend.py --max-images 200 --backends threshold,hybrid --skip-download
python scripts/score_external_labels.py --pred-dir path/to/cp_labels --max-images 50 --name cellprofiler
```

---

## What it contains

- Acquisition QC: focus, brightness, contrast, saturation, clipping, illumination, artifacts
- Segmentation: Otsu / adaptive / optional Cellpose / hybrid ensemble
- Validation metrics: IoU, Dice, instance F1 (centroid matching), count error
- Tracking, lineage, morphology/intensity/texture features
- Plate/well QC, Z-prime, replicate-aware statistics, provenance

Public API: `opticell`. No required GUI.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -e '.[cellpose]'   # optional
```

## Usage

```bash
opticell /path/to/images -o qc_summary.csv
python scripts/run_killer_workflow.py /path/to/images -o outputs/workflow_run --backend threshold
```

## Roadmap

See [`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md).

## License

AGPL-3.0-or-later. See `LICENSE`.
