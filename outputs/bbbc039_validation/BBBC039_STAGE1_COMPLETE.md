# BBBC039 Stage-1 complete (measured only)

All four backends scored with OptiCell `validation.py` on the same FOV set.
Empty-GT FOVs skipped (3 of 200) where noted.

## Full set (n=197 nonzero-GT)

| Backend | Dice | Instance F1 | Mean \|count err\| | Rel count |
|---------|-----:|------------:|------------------:|----------:|
| **OptiCell threshold** | 0.925 | **0.929** | **8.7** | **0.077** |
| **OptiCell hybrid** | 0.929 | 0.924 | 9.0 | 0.083 |
| **Cellpose-SAM (`cpsam`, GPU)** | **0.969** | 0.907 | 16.7 | 0.172 |
| **CellProfiler 4.2** | 0.895 | 0.722 | 28.1 | 0.281 |

Sources:
- threshold / hybrid: multi-backend n=200 runs (rel count on 197)
- `bbbc039_cellpose_n197.json` (Colab GPU, 2026-09-10)
- `bbbc039_cellprofiler_n197.json`

## Pilot n=50 (same ranking)

| Backend | Dice | Instance F1 | Mean \|count err\| |
|---------|-----:|------------:|------------------:|
| OptiCell threshold | 0.945 | **0.955** | **6.1** |
| OptiCell hybrid | 0.951 | 0.950 | 6.4 |
| Cellpose-SAM | **0.970** | 0.908 | 15.0 |
| CellProfiler | 0.897 | 0.722 | 26.2 |

## Ranking by use-case (BBBC039 fluorescent nuclei)

| Goal | Best measured |
|------|----------------|
| Instance identity / count accuracy | **OptiCell threshold** (then hybrid) |
| Pixel overlap (Dice/IoU) | **Cellpose-SAM** |
| Avoid large count bias | Prefer OptiCell classical/hybrid over CP or default cpsam |

Cellpose pattern is stable at scale: excellent Dice (~0.97), systematic over-count (rel ~0.17, F1 ~0.91).
CellProfiler (this export) trails on instance F1 and counts.

## Stage-1 checklist

| Backend | n=50 | n≈197–200 |
|---------|------|------------|
| OptiCell threshold | done | done |
| OptiCell hybrid | done | done |
| Cellpose-SAM | done | **done** |
| CellProfiler | done | done |

**Headline:** On BBBC039, OptiCell threshold/hybrid are **measurably best for instance F1 and cell counts**; Cellpose-SAM is **measurably best for pixel Dice**; this CellProfiler baseline is third on both axes.

---

*Stop rather than invent numbers.*
