# BBBC039 n=50 — full Stage-1 comparison (measured only)

**Policy:** every number below comes from a completed run on real BBBC039 images + official masks. No fabricated metrics.

| Field | Value |
|-------|-------|
| Dataset | BBBC039 (U2OS Hoechst nuclei) |
| n_images | **50** (same basename-sorted FOV list) |
| GT | Official instance masks |
| Metrics | OptiCell `validation.py` (identical code for all backends) |

## Aggregate metrics

| Backend | IoU | Dice | Instance F1 | Mean \|count err\| | Rel count err |
|---------|----:|-----:|------------:|------------------:|--------------:|
| **OptiCell threshold (Otsu)** | 0.8956 | **0.9448** | **0.9554** | **6.06** | **0.058** |
| **OptiCell hybrid (count-gated)** | 0.9078 | 0.9514 | 0.9500 | 6.42 | 0.065 |
| **Cellpose-SAM (`cpsam`)** | **0.9428** | **0.9705** | 0.9077 | 15.04 | — |
| **CellProfiler 4.2.x** | 0.8138 | 0.8971 | 0.7217 | 26.24 | 0.279 |

Sources:

- `bbbc039_threshold_n50.json` / export round-trip
- `bbbc039_hybrid_n50.json`
- `bbbc039_cellpose_n50.json`
- `bbbc039_cellprofiler_n50.json` ← **this run**

## Honest interpretation

1. **OptiCell classical threshold** is the strongest **count / instance-F1** baseline on this fluorescent nuclei subset (F1 0.955, mean abs count error ~6).
2. **Cellpose-SAM** leads **pixel overlap** (Dice 0.970) but over/under-segments more often (count error ~15, F1 0.908).
3. **Hybrid** stays near threshold on counts with a small Dice gain — a count-safe default, not dual-axis SOTA.
4. **CellProfiler** (this pipeline / export) underperforms OptiCell threshold on every headline metric here: lower Dice/IoU, much lower instance F1, ~4× count error. That is a **measured result for this configuration**, not a claim that “CellProfiler is always worse.” A differently tuned IdentifyPrimaryObjects pipeline could close the gap; re-score any new export with the same `score_external_labels.py`.

## What this answers (Stage 1)

> Is OptiCell measurably competitive with existing workflows on a fixed public set?

On BBBC039 n=50, with shared metrics code: **yes — OptiCell threshold/hybrid beat this CellProfiler export on instance F1 and count error; Cellpose still wins pure pixel overlap.**

## Not claimed

- Full BBBC039 n=200 for CellProfiler or Cellpose
- Runtime / GPU comparison (not logged in this paste)
- LIVECell / TissueNet / phase-contrast generalization

---

*Stop rather than invent numbers.*
