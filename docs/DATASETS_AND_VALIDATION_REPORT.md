# OptiCell — Large-Scale Public Datasets & Validation Policy

**Date:** 2026-08-30  
**Repository:** Virelion-Biotech/Virelion-OptiCell  
**Author note:** Prepared under explicit instruction to use real data only. No invented metrics, loss curves, or training results are reported here.

---

## 1. Policy (non-negotiable)

- OptiCell currently ships classical QC + optional Cellpose backends and model-agnostic validation metrics (`validation.py`: IoU, Dice, instance F1 via centroid matching, count error, etc.).
- There is **no** trained custom model or large-scale training run inside this repository as of this commit.
- Any future training or large-scale validation **must**:
  - Use only publicly documented or properly licensed data.
  - Keep train / validation / test splits disjoint.
  - Report only numbers that were actually computed on held-out data.
  - Stop rather than invent performance figures when data or compute are unavailable.
- This report deliberately contains **zero** fabricated accuracy / IoU / loss numbers.

---

## 2. Suitable public datasets (hundreds of thousands of cells or images)

The following are real, published resources that can support QC evaluation, segmentation benchmarking, or (with proper infrastructure) training. Sizes are taken from the original publications or official project pages.

### Segmentation / instance annotation (primary fit for OptiCell validation)

| Dataset | Approx. scale | Modality | Ground truth | Access / notes |
|---------|---------------|----------|--------------|----------------|
| **LIVECell** | 5,239 images; **>1.6 million** annotated cells (8 cell lines) | Phase-contrast (Incucyte) | Instance masks (COCO-style JSON) | https://sartorius-research.github.io/LIVECell/  · CC BY-NC 4.0 · AWS S3 + Figshare |
| **TissueNet** | ~2,600+ train images; **>1 million** whole-cell + nuclear annotations | Multiplex tissue (6 platforms, 9 organs) | Nuclear + whole-cell instance labels | https://datasets.deepcell.org/ (registration; non-commercial terms) |
| **CellFMCount** | 3,023 images; **>430,000** cell locations | Fluorescence immunocytochemistry | Dot annotations (counting) | Zenodo DOI 10.5281/zenodo.17088532 |
| **BBBC038** (2018 Data Science Bowl) | Diverse nuclei; tens of thousands of nuclei | Mixed fluorescence / histology | Instance masks | https://bbbc.broadinstitute.org/BBBC038 |
| **EVICAN** | ~4,600 images; ~26k–54k instances | Brightfield / phase | Cell + nucleus | Public; multi-line, multi-instrument |

### Focus / brightness / acquisition QC (direct fit for OptiCell QC flags)

| Dataset | Approx. scale | Relevance |
|---------|---------------|-----------|
| **BBBC006** | 768 fields × 32 z-planes (U2OS, Hoechst) | Explicit in-focus vs out-of-focus stacks — ideal for validating Laplacian / focus metrics |
| **BBBC005** | 9,600 fields (synthetic focus variation) | Controlled blur |
| AutoQC-Bench / related HT microscopy QC sets | Thousands of frames with annotated artifacts | Brightfield migration / anomaly QC |

### Very large imaging collections (profiling / phenotype; sparse or no dense instance GT)

| Collection | Scale | Notes |
|------------|-------|-------|
| **BBBC** overall | >11 million images across 50+ sets | https://bbbc.broadinstitute.org/ |
| **BBBC022 / BBBC036 / BBBC047** (Cell Painting) | Hundreds of thousands to millions of fields | Morphological profiling; limited dense instance GT |
| **JUMP-CP / IDR / HPA** | Hundreds of thousands to millions of images | Excellent for QC stress-testing (brightness, focus, artifacts); segmentation GT must be generated or obtained separately |
| Label-free bright-field ATOM collections | ~900k single-cell crops (classification) | Not full-field instance segmentation |

---

## 3. What OptiCell can do with these datasets today

Without additional compute or multi-TB storage in the present environment:

1. **Validation only (recommended next step)**  
   - Load a subset of LIVECell / BBBC038 / BBBC006 images + masks.  
   - Run existing OptiCell segmentation backends (threshold / adaptive / Cellpose if installed).  
   - Score with `paired_segmentation_metrics` / `benchmark_segmentation` from `validation.py`.  
   - Report real numbers only for the images actually processed.

2. **QC stress tests**  
   - BBBC006 (and similar) for focus / blur detection.  
   - Intensity statistics for brightness / saturation / clipping checks already present in the acquisition QC path.

3. **Training**  
   - Requires: local or cloud storage for the chosen corpus, GPU hours, and an explicit training module that does not yet exist in the repo.  
   - Cellpose already provides a strong generalist baseline; any new OptiCell-specific model must be trained and evaluated on held-out splits of the datasets above and must never claim performance that was not measured.

---

## 4. Immediate blockers for “hundreds of thousands of data” end-to-end training in this session

- Full LIVECell / TissueNet / Cell Painting downloads are multi-GB to multi-TB.  
- No GPU training infrastructure or multi-day job runner is attached to the current agent environment.  
- Therefore **no training was executed** and **no performance numbers are claimed**.

Any claim of “we trained on 250k images and achieved X” without the corresponding data hashes, code, and evaluation logs would violate the no-hallucination rule.

---

## 5. Concrete next actions (when data + compute are available)

1. Add a `datasets/` or `scripts/download_*.py` helper that records SHA-256 hashes of every downloaded archive.  
2. Write a thin adapter that converts LIVECell COCO masks / TissueNet labels into the label arrays expected by `validation.py`.  
3. Run a **subset** benchmark first (e.g., 100–500 images) and publish only those measured metrics.  
4. Scale only after the subset pipeline is deterministic and reproducible.  
5. Keep all evaluation tables under version control with the exact image list and software version.

---

## 6. References (public)

- LIVECell: Edlund et al., *Nat Methods* 2021 — https://sartorius-research.github.io/LIVECell/  
- TissueNet / Mesmer: Greenwald, Miller et al., *Nat Biotechnol* 2022 — https://datasets.deepcell.org/  
- Broad Bioimage Benchmark Collection — https://bbbc.broadinstitute.org/  
- CellFMCount — arXiv:2511.19351 / Zenodo  
- OptiCell validation API: `validation.py` in this repository

---

**Summary:** Massive real public datasets exist and are catalogued above. OptiCell already has the validation metrics needed to score segmentations on them. Full-scale training and the associated numerical results are **not** present in this commit because the data and compute required to produce them honestly are not available in the current session. Future work must start from real downloads and measured numbers only.
