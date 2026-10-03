# Stage 2 — CTC TRA / DET / LNK (measured)

**Method:** OptiCell threshold segmentation + assignment tracking (`max_distance=50`, `max_gap=1`)  
**Export:** continuous tracklets via `scripts/export_ctc_res.py`  
**Scoring:** [`traccuracy`](https://github.com/live-image-tracking-tools/traccuracy) `CTCMatcher` + `CTCMetrics`  
**Policy:** numbers below are from completed Colab runs only. No invented TRA.

## Results table

| Dataset | Seq | TRA | DET | LNK | AOGM | fn_nodes | fp_nodes | fn_edges | fp_edges |
|---------|-----|----:|----:|----:|-----:|---------:|---------:|---------:|---------:|
| Fluo-N2DH-GOWT1 | 01 | 0.364 | 0.367 | 0.346 | 15022 | 1134 | 1687 | 1330 | 0 |
| Fluo-N2DH-GOWT1 | 02 | 0.310 | 0.309 | 0.314 | 19939 | 1543 | 1966 | 1695 | 0 |
| Fluo-N2DH-SIM+ | 01 | **0.809** | **0.831** | **0.664** | 5705 | 384 | 533 | 854 | 2 |
| Fluo-N2DH-SIM+ | 02 | **0.000** | **0.000** | 0.098 | 59628 | 2748 | **27547** | 3023 | 0 |
| Fluo-N2DL-HeLa | 01 | 0.652 | 0.684 | 0.439 | 34538 | 2440 | 914 | 4717 | 73 |
| Fluo-N2DL-HeLa | 02 | 0.715 | 0.754 | 0.453 | 83179 | 5251 | 2761 | 13480 | 309 |

Artifact: `tra_aggregate.csv` + per-sequence `*_tra.json` from Colab.

## Interpretation (honest)

1. **Best sequence:** SIM+ / 01 — TRA ~0.81, DET ~0.83. Simulated nuclei suit simple thresholding.
2. **HeLa:** mid TRA (0.65–0.72); linking (LNK ~0.44–0.45) is the weak leg — many FN edges, some NS (non-split / division-related) errors. OptiCell linker does **not** model divisions (parent=0).
3. **GOWT1:** low TRA/DET (~0.31–0.36). Threshold over/under-segments GFP stem-cell nuclei vs GT; high FP+FN nodes dominate.
4. **SIM+ / 02 failure:** TRA=0, DET=0, **fp_nodes=27,547**. Classic threshold collapse (noise fragments or intensity mismatch on that sequence). Reported as measured failure — not hidden.

## What this does **not** claim

- Competition-level CTB ranking (top methods are typically deep learning + division-aware linking).
- Cellpose/hybrid TRA on these sequences (not run yet).
- SEG metric (would need CTC SEG GT path separately).

## Next measured steps (optional)

| Priority | Action |
|----------|--------|
| 1 | Re-run SIM+ / 02 + GOWT1 with **Cellpose** backend for TRA (expect DET lift) |
| 2 | Add division-aware linking or accept LNK ceiling without it |
| 3 | Stage 3: auto backend selection when DET/confidence is poor |

---

*Stop rather than invent track accuracy. Cite Cell Tracking Challenge if publishing.*
