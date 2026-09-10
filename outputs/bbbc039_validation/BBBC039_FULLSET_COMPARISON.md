# BBBC039 full set — Stage-1 comparison (measured only)

**Policy:** numbers from completed runs only. Empty-GT FOVs excluded where noted (3 of 200).

## n ≈ 200 (nonzero-GT FOVs)

| Backend | n | Dice | Instance F1 | Mean \|count err\| | Rel count err |
|---------|--:|-----:|------------:|------------------:|--------------:|
| **OptiCell threshold** | 197* | **0.925** | **0.929** | **8.74** | **0.077** |
| **OptiCell hybrid** | 197* | **0.929** | 0.924 | 8.97 | 0.083 |
| **CellProfiler 4.2** | **197** | 0.895 | 0.722 | 28.12 | 0.281 |

\* OptiCell n=200 runs reported means over the full request; relative count used 197 nonzero-GT FOVs. CP JSON explicitly `n_images=197`.

Sources:
- `bbbc039_threshold_n200.json` / multi-backend report
- hybrid n=200 multi-backend
- `bbbc039_cellprofiler_n197.json` ← **this run**

## n=50 pilot (includes Cellpose)

| Backend | Dice | Instance F1 | Mean \|count err\| |
|---------|-----:|------------:|------------------:|
| OptiCell threshold | 0.945 | **0.955** | **6.1** |
| OptiCell hybrid | 0.951 | 0.950 | 6.4 |
| Cellpose-SAM | **0.970** | 0.908 | 15.0 |
| CellProfiler | 0.897 | 0.722 | 26.2 |

## Interpretation (measured, this configuration)

1. **OptiCell threshold/hybrid** remain strongest on **instance F1 and counts** at full BBBC039 scale vs this CellProfiler export (~0.93 F1 vs ~0.72; count error ~9 vs ~28).
2. CP Dice (~0.90) is closer to OptiCell pixel overlap than its instance F1 — over-segmentation / split-merge likely (low precision 0.65, higher recall 0.82).
3. **Cellpose n=200** still not measured (GPU optional).
4. Results describe **this** CP pipeline/export, not every possible CellProfiler tuning.

## Stage-1 status

| Comparator | n=50 | n≈200 |
|------------|------|-------|
| OptiCell threshold | done | done |
| OptiCell hybrid | done | done |
| Cellpose-SAM | done | pending GPU |
| CellProfiler | done | **done** |

Headline: on BBBC039 fluorescent nuclei, OptiCell classical/hybrid backends are **measurably better on instance metrics and counts** than this CellProfiler baseline, at pilot and full-set scale.

---

*Stop rather than invent numbers.*
