# CellProfiler baseline on BBBC039 (Stage 1)

**Goal:** fair comparison on the **same FOV list** used by OptiCell threshold / hybrid / Cellpose.

Policy: publish only measured metrics from this procedure.

## Why

The product headline is: *Is OptiCell measurably better, faster, or more reproducible than existing workflows?*

CellProfiler is the established open platform for HCS. BBBC006 itself used CellProfiler `IdentifyPrimaryObjects` (Otsu 2-class) for nuclei counts.

## Procedure

### 1. Install CellProfiler

https://cellprofiler.org/releases (GUI or headless `cellprofiler` CLI).

### 2. Input images

Use the same BBBC039 TIFFs OptiCell scores:

```text
data/bbbc039/images/images/*.tif
```

Basename order must match `scripts/run_bbbc039_validation.py` (sorted stems).

### 3. Recommended modules (nuclei, fluorescent)

Document these settings in your run notes:

| Module | Setting |
|--------|---------|
| Images / Metadata / NamesAndTypes | Load grayscale DNA/Hoechst channel |
| IdentifyPrimaryObjects | Input = DNA |
| Typical diameter | min 10–15 px, max ~80–120 px (tune on 1 FOV) |
| Threshold | Global **Otsu**, **two-class**, minimize weighted variance |
| Discard objects outside diameter | Yes |
| Discard objects touching border | Optional (report which) |
| Declump | Shape or Intensity; report choice |
| ConvertObjectsToImage | Objects → uint16 label image |
| SaveImages | One label PNG/TIFF per FOV, stem = image stem |

Optional reference pipeline style: BBBC006 `Batch_data.cppipe`  
https://data.broadinstitute.org/bbbc/BBBC006/Batch_data.cppipe

### 4. Score with OptiCell (same metrics)

```bash
python scripts/score_external_labels.py \
  --pred-dir path/to/cellprofiler_labels \
  --data-dir data/bbbc039 \
  --max-images 50 \
  --name cellprofiler \
  --out-dir outputs/bbbc039_validation
```

Writes `bbbc039_cellprofiler_nN.json` with IoU, Dice, instance F1, count error — **identical metric code** as OptiCell backends.

### 5. Report

Paste SUMMARY into a PR / `outputs/bbbc039_validation/BBBC039_*_CELLPROFILER.md` with:

- CellProfiler version
- Exact module settings
- n FOVs and image list hash if available
- Runtime wall-clock

Do **not** invent numbers. If CP is not run, leave the row blank.

## OptiCell numbers already measured (same set)

| Backend | n | Dice | F1 | \|count err\| |
|---------|--:|-----:|---:|-------------:|
| Threshold | 50 | 0.945 | 0.955 | 6.1 |
| Cellpose-SAM | 50 | 0.970 | 0.908 | 15.0 |
| Hybrid | 50 | 0.951 | 0.950 | 6.4 |
| Threshold | 200 | 0.925 | 0.929 | 8.7 |
| Hybrid | 200 | 0.929 | 0.924 | 9.0 |

Fill CellProfiler when the export + score step completes.
