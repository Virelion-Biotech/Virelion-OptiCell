# Stage 2 — CTC time-lapse (measured)

**Pipeline:** QC → threshold segment → features → rule phenotype → assignment tracking  
**Backend:** OptiCell threshold  
**Tracking:** `max_distance_px=50`, `max_gap=1`  
**Source data:** Cell Tracking Challenge training sets ([cite CTC](https://celltrackingchallenge.net/datasets/))

Policy: numbers below are from completed Colab runs only. **No TRA accuracy vs GT** is claimed here (that needs a separate CTC TRA scorer).

## Aggregate (all six sequences)

| Run | Frames | Mean count/frame | Objects total | Tracks | Mean focus | Mean conf |
|-----|-------:|-----------------:|--------------:|-------:|-----------:|----------:|
| Fluo-N2DH-GOWT1 / 01 | 92 | 28.4 | 2,611 | 730 | 274 | 100 |
| Fluo-N2DH-GOWT1 / 02 | 92 | 32.0 | 2,940 | 869 | 112 | 100 |
| Fluo-N2DH-SIM+ / 01 | 65 | 42.3 | 2,749 | 596 | 2,279 | 100 |
| Fluo-N2DH-SIM+ / 02 | 150 | 187.7 | 28,157 | 8,013 | 11,643 | 100 |
| Fluo-N2DL-HeLa / 01 | 92 | 72.6 | 6,679 | 1,611 | 79 | 100 |
| Fluo-N2DL-HeLa / 02 | 92 | 229.6 | 21,119 | 5,464 | 112 | 100 |

**Totals (measured):** 583 frames · 64,255 object detections · 17,283 track IDs initiated.

Artifact: `stage2_ctc_aggregate.csv`.

## What this demonstrates

1. End-to-end Stage-2 workflow runs on **real public time-lapse** (not BBBC039 wells).
2. Tracking completes without LAP crashes after the infeasible-cost fix.
3. Confidence stayed at 100 on these fluorescent sequences with the threshold backend (QC flags quiet).

## What this does **not** claim

- **Track accuracy (TRA)** vs CTC `*_GT/TRA` — not computed.
- **Segmentation accuracy (SEG)** vs CTC GT masks — not computed in this pass.
- Phenotype `positive_fraction=0.0` in the CSV is a **label-name quirk** (`pass_rules` vs expected `positive` in `group_phenotype_summary`); mean rule scores were non-zero in JSON summaries (e.g. GOWT1_01 ~1.96 / 3). Treat phenotype column as incomplete until the summary key is aligned.

## Next (optional, measured only)

1. Fix phenotype summary label to match `score_cells` output labels.
2. Add CTC TRA/SEG scoring script against `01_GT` when ready — publish only real TRA/SEG.
3. Stage 3: confidence-gated backend selection on these same sequences.

---

*Stop rather than invent track accuracy.*
