# BBBC039 empty ground-truth FOVs (measured discovery)

During n=200 validation, **3 / 200** paired FOVs decoded to `truth_count == 0` while backends predicted nuclei. That produced `relative_count_error: inf` in aggregate means before the finite-mean fix.

## FOVs (basename)

| Image | pred (threshold/hybrid) |
|-------|------------------------:|
| `IXMtest_F13_s7_w13C1B1D8C-293E-454F-B0FD-6C2C3F9F5173.tif` | 28 |
| `IXMtest_L01_s2_w1E5038251-DBA3-44D0-BC37-E43E2FC8C174.tif` | 68 |
| `IXMtest_L10_s6_w12D12D64C-2639-4CA8-9BB4-99F92C9B7068.tif` | 40 |

Same three for both `bbbc039_threshold_n200.json` and `bbbc039_hybrid_n200.json`.

## Likely causes

1. Official mask PNG is all background for that FOV (annotation gap), or
2. Color encoding not captured by channel-0 / unique-color decode for those files.

Official decode pattern (Caicedo gist): RGB channel 0 → connected components.

## Handling (as of this commit)

- `validation.count_error`: empty GT + pred>0 → `relative_count_error = nan` (not `inf`).
- `benchmark_segmentation`: means use **finite values only** (`nanmean`-style).
- Runner should log empty-GT FOVs; IoU/Dice/F1/absolute count error remain defined.

## Recompute headline relative error

After `git pull`, re-run summary on existing JSON or re-score n=200. Expected: finite `relative_count_error_mean` over the 197 FOVs with non-empty GT.

---

*Measured discovery — not a claim that the three FOVs lack nuclei in the raw TIFF.*
