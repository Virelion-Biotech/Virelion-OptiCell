# BBBC039 n=50 — threshold vs Cellpose vs hybrid (measured only)

**Policy:** numbers below were computed on real BBBC039 images + official masks. No fabricated metrics.

| Field | Value |
|-------|-------|
| Dataset | BBBC039 (U2OS Hoechst nuclei) |
| Source | https://bbbc.broadinstitute.org/BBBC039 |
| n_images | **50** (same FOV subset for all backends) |
| Paired corpus available | 200/200 |
| Script | `scripts/run_bbbc039_validation.py` |

## Aggregate metrics (measured)

| Metric | Threshold (Otsu) | Cellpose (`cpsam`) | Hybrid (count-gated) |
|--------|-----------------:|-------------------:|---------------------:|
| Pixel IoU | 0.8956 | **0.9428** | 0.9078 |
| Pixel Dice | 0.9448 | **0.9705** | 0.9514 |
| Instance F1 | **0.9554** | 0.9077 | 0.9500 |
| Mean \|count error\| | **6.06** | 15.04 | 6.42 |

Source files on the runner:

- `bbbc039_threshold_n50.json`
- `bbbc039_cellpose_n50.json`
- `bbbc039_hybrid_n50.json`

## Honest interpretation

- **Cellpose** leads on pixel overlap (Dice/IoU).
- **Threshold** leads on instance F1 and count error.
- **Hybrid** (prefer Cellpose when counts agree, else threshold) lands near threshold on counts/F1 and slightly above threshold on Dice — a **count-safe default**, not yet a dual-axis winner.
- Next measured step for hybrid: watershed split of Cellpose blobs using threshold seeds (see roadmap P1), then re-score the same 50 FOVs.
- None of these are claims about full BBBC039 (200), LIVECell, or CellProfiler comparisons.

---

*Stop rather than invent numbers.*
