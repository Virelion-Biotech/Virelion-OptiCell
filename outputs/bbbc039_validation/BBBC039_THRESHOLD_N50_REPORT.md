# BBBC039 validation report — OptiCell threshold backend

**Measured only. No fabricated metrics.**

| Field | Value |
|-------|-------|
| Dataset | BBBC039 (U2OS Hoechst nuclei) |
| Source | https://bbbc.broadinstitute.org/BBBC039 |
| Backend | OptiCell classical Otsu threshold (`segment_threshold`) |
| n_images | **50** (first 50 paired FOVs after basename sort) |
| Paired corpus | 200/200 images with masks available |
| Script | `scripts/run_bbbc039_validation.py` |
| Run date | 2026-09-07 |
| Host | local (user machine) |

## Mask decode check

```
[mask-check] IXMtest_A02_s1_...png: shape=(520, 696) max_label=94 fg_pixels=70682
```

Ground-truth instance labels loaded correctly after RGB-aware decode.

## Aggregate metrics (n=50)

| Metric | Value |
|--------|------:|
| pixel IoU (mean) | 0.8956 |
| pixel Dice (mean) | 0.9448 |
| pixel precision (mean) | 0.9837 |
| pixel recall (mean) | 0.9294 |
| instance F1 (mean) | 0.9554 |
| instance precision (mean) | 0.9837 |
| instance recall (mean) | 0.9294 |
| absolute count error (mean) | 6.06 |
| relative count error (mean) | 0.0583 |

## Interpretation (honest)

- Classical Otsu + morphology is a **strong baseline** on this fluorescent nuclei subset: Dice ≈ 0.94, instance F1 ≈ 0.96.
- Systematic under-count (mean abs error ≈ 6 nuclei/FOV; relative ≈ 5.8%) — typical when touching nuclei merge under a single threshold.
- This is **not** a claim about LIVECell, TissueNet, phase-contrast, or full BBBC039 (200). It is only the 50 FOVs scored in this run.
- TIFF OpenCV warnings (tags 317 / 33628) are metadata noise and did not block reads.

## Artifacts on runner machine

- `outputs/bbbc039_validation/bbbc039_threshold_n50.json`
- `outputs/bbbc039_validation/bbbc039_threshold_n50.csv`

## Next measured steps (when run)

1. Full BBBC039: `--max-images 200`
2. Same 50 with `--backend cellpose` (nuclei model) for comparison
3. Optional adaptive threshold
4. Only then consider larger corpora (LIVECell) with the same script pattern

---

*Policy: stop rather than invent numbers. All values above were computed on real BBBC039 images + official masks.*
