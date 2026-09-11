# Stage 3 — Multi-backend orchestration + TRA (measured)

**Backends:** threshold, hybrid, Cellpose-SAM (`cpsam`, GPU)  
**Tracking:** assignment, `max_distance=50`, `max_gap=1`  
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

## Auto-select v1 vs best TRA (original runs)

| Seq | Auto v1 | Auto TRA | Best | Best TRA | Match? |
|-----|---------|---------:|------|---------:|:------:|
| GOWT1 01 | cellpose | 0.973 | cellpose | 0.973 | Yes |
| GOWT1 02 | cellpose | 0.918 | cellpose | 0.918 | Yes |
| SIM+ 01 | threshold | 0.809 | cellpose | 0.957 | **No** |
| SIM+ 02 | cellpose | 0.505 | cellpose | 0.505 | Yes |
| HeLa 01 | cellpose | 0.914 | cellpose | 0.914 | Yes |
| HeLa 02 | threshold | 0.715 | cellpose | 0.925 | **No** |

v1 match rate: **4 / 6**. Failure mode: `mean_confidence=100` for all backends, so v1 ranked by lowest `count_cv` and preferred threshold.

## Selection rule v2 (code fix)

```
1. Reject: failed runs, mean_count==0, count_cv > 1.0, zero_object_fraction > 0.25
2. Prefer: cellpose > hybrid > threshold among survivors
3. Tie-break: lower count_cv, higher confidence, more frames
```

Offline validation on the **same measured CSV** (`scripts/validate_stage3_selection.py`):

| Seq | v2 pick | Best TRA backend | Match? |
|-----|---------|------------------|:------:|
| GOWT1 01/02 | cellpose | cellpose | Yes |
| SIM+ 01 | cellpose | cellpose | Yes |
| SIM+ 02 | cellpose (thr/hybrid rejected cv>1) | cellpose | Yes |
| HeLa 01/02 | cellpose | cellpose | Yes |

**v2 match rate on measured data: 6 / 6** (no new GPU runs required).

## Hard truths

1. **Cellpose is the real TRA engine** on these fluorescent CTC sets.
2. **FOV confidence saturated at 100** — not a selector by itself.
3. **count_cv > 1** flags threshold/hybrid collapse (SIM+02).
4. Hybrid ≈ threshold on HeLa in these runs; helps GOWT1 vs threshold only.

## Production default

```
if cellpose available and not rejected by count_cv / zero-count:
    use cellpose
elif hybrid available and stable:
    use hybrid
else:
    use threshold with collapse guards
```

## Artifacts

- `stage3_tra_by_backend.csv`
- `stage3_autoselect_vs_best.csv` (v1 live run)
- `scripts/run_stage3_orchestrate.py` (v2 rule)
- `scripts/validate_stage3_selection.py`

---

*Stop rather than invent scores.*
