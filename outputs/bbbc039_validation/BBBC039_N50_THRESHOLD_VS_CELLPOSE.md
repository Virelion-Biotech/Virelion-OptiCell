# BBBC039 n=50 — threshold vs Cellpose (measured only)

**Policy:** numbers below were computed on real BBBC039 images + official masks. No fabricated metrics.

| Field | Value |
|-------|-------|
| Dataset | BBBC039 (U2OS Hoechst nuclei) |
| Source | https://bbbc.broadinstitute.org/BBBC039 |
| n_images | **50** (same FOV subset for both backends) |
| Paired corpus available | 200/200 |
| Script | `scripts/run_bbbc039_validation.py` |

## Aggregate metrics (measured)

| Metric | Threshold (Otsu) | Cellpose (`cpsam`) |
|--------|-----------------:|-------------------:|
| Pixel IoU | 0.8956 | **0.9428** |
| Pixel Dice | 0.9448 | **0.9705** |
| Instance F1 | **0.9554** | 0.9077 |
| Mean \|count error\| | **6.06** | 15.04 |

Source files on the runner:

- `outputs/bbbc039_validation/bbbc039_threshold_n50.json`
- `outputs/bbbc039_validation/bbbc039_cellpose_n50.json`

## Honest interpretation

- **Pixel overlap:** Cellpose-SAM improves IoU (+0.047) and Dice (+0.026) vs classical Otsu on this fluorescent nuclei subset.
- **Instance matching / counts:** Threshold has higher instance F1 and lower absolute count error. Cellpose under/over-segments more often on this 50-FOV sample (mean \|count err\| ≈ 15 vs ≈ 6).
- Neither result is a claim about full BBBC039 (200), LIVECell, TissueNet, or other modalities.
- TIFF OpenCV tag warnings remain harmless metadata noise.

## Next measured steps (optional)

1. Same comparison on `--max-images 200`
2. Adaptive threshold baseline
3. Commit JSON/CSV artifacts alongside this report when available

---

*Stop rather than invent numbers. All values above are from completed local/Colab runs.*
