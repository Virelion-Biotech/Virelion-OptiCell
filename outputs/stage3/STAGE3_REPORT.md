# Stage 3 — Multi-backend orchestration + TRA (measured)

**Backends:** threshold, hybrid, Cellpose-SAM (`cpsam`, GPU)  
**Tracking:** assignment, `max_distance=50`, `max_gap=1`  
**Select rule:** higher mean FOV confidence → lower count CV → fewer low-conf FOVs  
**TRA:** traccuracy CTCMetrics  
**Policy:** measured only.

## TRA by backend

| Dataset | Seq | threshold | hybrid | **cellpose** |
|---------|-----|----------:|-------:|-------------:|
| GOWT1 | 01 | 0.364 | 0.598 | **0.973** |
| GOWT1 | 02 | 0.310 | 0.778 | **0.918** |
| SIM+ | 01 | 0.809 | 0.949 | **0.957** |
| SIM+ | 02 | 0.000 | 0.000 | **0.505** |
| HeLa | 01 | 0.652 | 0.652 | **0.914** |
| HeLa | 02 | 0.715 | 0.715 | **0.925** |

Cellpose wins TRA on **all six** sequences.

## Auto-select vs best TRA

| Seq | Auto-selected | Auto TRA | Best backend | Best TRA | Match? |
|-----|---------------|---------:|--------------|---------:|:------:|
| GOWT1 01 | cellpose | 0.973 | cellpose | 0.973 | Yes |
| GOWT1 02 | cellpose | 0.918 | cellpose | 0.918 | Yes |
| SIM+ 01 | threshold | 0.809 | cellpose | 0.957 | **No** |
| SIM+ 02 | cellpose | 0.505 | cellpose | 0.505 | Yes |
| HeLa 01 | cellpose | 0.914 | cellpose | 0.914 | Yes |
| HeLa 02 | threshold | 0.715 | cellpose | 0.925 | **No** |

**Match rate: 4 / 6.**

## Hard truths

1. **Cellpose is the real TRA engine** on these fluorescent CTC sets. Threshold alone is not competition-grade tracking.
2. **FOV confidence is saturated at 100** on every backend here — it cannot break ties. Selection fell through to **count_cv**, which picked threshold on SIM+01 and HeLa02 (lower CV) and **missed** the best TRA.
3. **SIM+02** remains hard: threshold/hybrid TRA=0 (massive FP); Cellpose recovers to TRA=0.50 — useful, not solved.
4. Hybrid ≈ threshold on HeLa (same masks path in practice for those runs); hybrid helps GOWT1 vs pure threshold but loses to Cellpose.

## Product implication

Default production path should be:

```
prefer Cellpose when GPU available
else hybrid/threshold with hard guards on count_cv / FP explosion
```

Improve Stage-3 selection by **not** relying on confidence alone when all scores are 100. Use count_cv **plus** a cheap DET proxy or prefer Cellpose when installed.

## Artifacts

- `stage3_tra_by_backend.csv`
- `stage3_autoselect_vs_best.csv`

Big mask zips are optional provenance, not required for the metric claim.

---

*Stop rather than invent scores.*
