# Virelion-OptiCell

**Headless Python toolkit** for microscopy QC, segmentation, tracking, phenotyping, and experiment-level quantitative analysis.

**Product thesis:** given raw microscopy data, OptiCell should produce *trustworthy* measurements with less manual work and explicit QC — not another black-box segmenter.

We publish **measured** numbers only. No fabricated benchmarks.

---

## Measured baseline — BBBC039 (U2OS Hoechst nuclei)

Public dataset: [BBBC039](https://bbbc.broadinstitute.org/BBBC039) · 200 FOVs with instance masks.

| Backend | n | Dice | Instance F1 | Mean \|count err\| | Rel. count err. |
|---------|--:|-----:|------------:|------------------:|----------------:|
| **Threshold (Otsu)** | 50 | 0.945 | **0.955** | **6.1** | 0.058 |
| **Cellpose-SAM (`cpsam`)** | 50 | **0.970** | 0.908 | 15.0 | — |
| **Hybrid (count-gated)** | 50 | 0.951 | 0.950 | 6.4 | 0.065 |
| **Threshold** | **200** | 0.925 | **0.929** | **8.7** | **0.077** |
| **Hybrid** | **200** | **0.929** | 0.924 | 9.0 | 0.083 |

- n=200 relative count error uses **197 FOVs** with nonzero ground truth (3 empty-GT masks documented under `outputs/bbbc039_validation/`).
- Hybrid prefers Cellpose when object counts agree with threshold; otherwise threshold (count-safe default).
- Full write-ups: `outputs/bbbc039_validation/`.

```bash
# Reproduce (data cached after first download)
pip install -e '.[cellpose]'   # optional for Cellpose/hybrid
python scripts/run_bbbc039_multi_backend.py --max-images 200 --backends threshold,hybrid --skip-download
```

---

## What it contains

- Acquisition QC: focus, brightness, contrast, saturation, clipping, illumination, artifacts
- Segmentation: Otsu / adaptive / optional Cellpose / **hybrid** ensemble
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
# optional
pip install -e '.[cellpose]'
pip install packaging   # needed by some Cellpose installs
```

## Usage

```bash
opticell /path/to/images -o qc_summary.csv
opticell /path/to/images --cell-method cellpose -o qc_summary.csv
```

```python
from opticell import analyze_folder
result = analyze_folder("images/")
```

### Benchmark a public set

```bash
python scripts/run_bbbc039_validation.py --max-images 50 --backend threshold --skip-download
python scripts/run_bbbc039_validation.py --max-images 50 --backend hybrid --gpu --skip-download
python scripts/run_bbbc039_multi_backend.py --max-images 200 --backends threshold,hybrid --skip-download
```

## Roadmap

See [`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md): Stage 1 scientific benchmark → one killer workflow (QC→segment→track→phenotype) → orchestration AI → usability → hard real-world datasets.

## Limitations

QC gates are decision aids, not biological proof. Segmentation quality depends on modality and parameters. Cellpose depends on the chosen model and domain. Always keep train/val/test discipline if you fine-tune.

## License

AGPL-3.0-or-later. See `LICENSE`.
