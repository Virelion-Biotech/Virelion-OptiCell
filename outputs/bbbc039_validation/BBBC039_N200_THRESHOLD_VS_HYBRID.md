# BBBC039 n=200 — threshold vs hybrid (measured only)

**Policy:** values below are from completed runs on real BBBC039 images + official masks. No fabricated metrics.

| Field | Value |
|-------|-------|
| Dataset | BBBC039 (U2OS Hoechst nuclei) |
| n_images scored | **200** paired FOVs |
| n with nonzero GT | **197** (3 empty-GT FOVs excluded from relative count mean) |
| Backends | threshold (Otsu), hybrid (count-gated) |
| Script | `scripts/run_bbbc039_multi_backend.py` |

## Aggregate metrics (measured)

| Metric | Threshold | Hybrid |
|--------|----------:|-------:|
| Pixel IoU | 0.8777 | **0.8860** |
| Pixel Dice | 0.9249 | **0.9294** |
| Instance F1 | **0.9289** | 0.9239 |
| Instance precision | 0.9680 | 0.9517 |
| Instance recall | 0.9121 | 0.9187 |
| Mean \|count error\| | **8.74** | 8.97 |
| Mean relative count error (197 FOVs) | **0.0770** | 0.0827 |

Empty-GT FOVs (documented in `BBBC039_EMPTY_GT_FOVS.md`):

- `IXMtest_F13_s7_w13C1B1D8C-293E-454F-B0FD-6C2C3F9F5173`
- `IXMtest_L01_s2_w1E5038251-DBA3-44D0-BC37-E43E2FC8C174`
- `IXMtest_L10_s6_w12D12D64C-2639-4CA8-9BB4-99F92C9B7068`

## vs n=50 pilot (measured earlier)

| Backend | n | Dice | F1 | \|count err\| | rel count |
|---------|--:|-----:|---:|-------------:|----------:|
| Threshold | 50 | 0.9448 | 0.9554 | 6.06 | 0.058 |
| Threshold | 200 | 0.9249 | 0.9289 | 8.74 | **0.077** |
| Hybrid | 50 | 0.9514 | 0.9500 | 6.42 | 0.065 |
| Hybrid | 200 | 0.9294 | 0.9239 | 8.97 | **0.083** |

Full corpus is harder than the first-50 slice.

## Honest interpretation

- **Hybrid** remains a mild Dice/IoU improvement over threshold with nearly the same count error and slightly lower instance F1.
- **Threshold** remains the count/F1-safer classical default.
- Relative count error is finite once empty-GT FOVs are handled (`nan` + finite mean).
- Cellpose n=200 still pending GPU run.

---

*Stop rather than invent numbers.*
