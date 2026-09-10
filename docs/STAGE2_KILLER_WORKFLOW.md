# Stage 2 — Killer workflow

**Goal:** one command that turns a folder of microscopy images into QC + segmentation + per-cell features + explicit phenotype calls, with optional tracking on real time-lapse data.

```
images → QC (focus/brightness/sat) → segment → confidence
      → object features → rule-based phenotype
      [→ track link + trajectory summary if --enable-tracking]
```

## Run (laptop, no GPU)

```bash
cd ~/Virelion-OptiCell
git pull origin main
source .venv/bin/activate

# BBBC039 FOVs — phenotype only (NOT time-lapse; do not enable tracking)
python scripts/run_killer_workflow.py data/bbbc039/images/images \
  -o outputs/stage2_bbbc039_threshold \
  --backend threshold \
  --max-images 30
```

Outputs under `outputs/stage2_bbbc039_threshold/`:

| File | Content |
|------|---------|
| `workflow_summary.json/csv` | Per-FOV QC + counts + confidence |
| `masks/*_labels.png` | Instance label maps |
| `cell_features_phenotype.csv` | Per-object morphology + intensity + rule scores |
| `phenotype_summary.csv` | Aggregate positive fraction |

## Time-lapse tracking

Only when frames are **the same field**, ordered by filename:

```bash
python scripts/run_killer_workflow.py /path/to/timelapse_frames \
  -o outputs/stage2_timelapse \
  --backend threshold \
  --enable-tracking \
  --track-max-distance 30 \
  --track-max-gap 1
```

Adds `tracks.csv` and `track_summary.csv` (path length, speed, straightness).

**Do not** pass `--enable-tracking` on BBBC039 multi-well images — links would be meaningless.

## Phenotype rules (explicit, auditable)

Default rules in `run_killer_workflow.py`:

| Feature | Rule |
|---------|------|
| `area_px` | ≥ 50 |
| `circularity` | ≥ 0.4 |
| `mean_intensity` | ≥ 40 |

Edit rules in code or fork `phenotype.Rule` / `score_cells` for assay-specific cutoffs. No black-box classifier.

## Relation to Stage 1

Stage 1 proved segmentation quality on BBBC039 vs Cellpose/CellProfiler.
Stage 2 uses the **same** segmentation backends to produce **assay-facing tables** (features, phenotype, optional tracks).

## Status

| Piece | Status |
|-------|--------|
| QC + segment + confidence | Done |
| Object features | Done |
| Rule phenotype | Done |
| Tracking (optional TL) | Done (needs real TL data to evaluate) |
| HITL / auto backend | Stage 3 |

---

*Measured outputs only. Stop rather than invent track accuracy on non-TL data.*
