# BBBC039 n=200 — threshold vs hybrid (measured only)

**Policy:** values below are from completed runs on real BBBC039 images + official masks. No fabricated metrics.

| Field | Value |
|-------|-------|
| Dataset | BBBC039 (U2OS Hoechst nuclei) |
| n_images | **200** (full paired corpus) |
| Backends | threshold (Otsu), hybrid (count-gated) |
| Script | `scripts/run_bbbc039_multi_backend.py` / `run_bbbc039_validation.py` |

## Aggregate metrics (measured)

Backend order matches the user’s paste: first SUMMARY = **threshold**, second = **hybrid** (higher Dice / IoU, slightly higher count error — same pattern as n=50).

| Metric | Threshold | Hybrid |
|--------|----------:|-------:|
| Pixel IoU | 0.8777 | **0.8860** |
| Pixel Dice | 0.9249 | **0.9294** |
| Instance F1 | **0.9289** | 0.9239 |
| Instance precision | 0.9680 | 0.9517 |
| Instance recall | 0.9121 | 0.9187 |
| Mean \|count error\| | **8.74** | 8.97 |
| Relative count error (mean) | **inf** | **inf** |

## Comparison to n=50 (same backends, measured earlier)

| Backend | n | Dice | F1 | \|count err\| |
|---------|--:|-----:|---:|-------------:|
| Threshold | 50 | 0.9448 | 0.9554 | 6.06 |
| Threshold | 200 | 0.9249 | 0.9289 | 8.74 |
| Hybrid | 50 | 0.9514 | 0.9500 | 6.42 |
| Hybrid | 200 | 0.9294 | 0.9239 | 8.97 |

Full-corpus scores are **lower** than the first-50 subset — expected; the pilot FOVs were not a random harder tail.

## `relative_count_error: inf` (important)

`validation.count_error` returns `inf` when `truth_count == 0` and `pred_count > 0`. A mean of **inf** means **at least one FOV** in the 200 has an empty decoded ground-truth mask (or max label 0) while the backend predicted objects.

This does **not** invalidate IoU/Dice/F1/absolute count error (those are finite). It does mean:

1. Relative count error must not be quoted as a single finite headline number for n=200 until empty-GT FOVs are filtered or fixed.
2. Follow-up: list FOVs with `truth_count == 0` from the per-image CSV and verify mask decode / pairing.

## Honest interpretation

- On the **full 200**, hybrid still looks like a mild Dice/IoU gain over threshold with nearly the same count error and slightly lower instance F1 — consistent with n=50.
- Neither backend claims SOTA; both are strong classical / hybrid baselines for fluorescent nuclei QC.
- Cellpose n=200 not included in this paste (GPU optional next).

## Next measured steps

1. Filter or fix empty-GT FOVs; recompute finite `relative_count_error_mean`.
2. Run cellpose n=200 when GPU available; extend comparison table.
3. CellProfiler baseline on the same basename-sorted list.

---

*Stop rather than invent numbers.*
