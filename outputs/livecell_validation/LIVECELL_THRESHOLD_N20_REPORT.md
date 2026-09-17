# LIVECell validation — threshold / Otsu (measured)

**Dataset:** [LIVECell](https://github.com/sartorius-research/LIVECell) (CC BY-NC 4.0, non-commercial)  
**Split:** val · **Backend:** threshold · **GPU:** false  
**n_scored:** 20 · **mean / max GT cells per image:** 197.6 / 425  
**Timestamp (UTC):** 2026-09-17T15:02:55.342929+00:00  
**Same FOV list as** `livecell_val_cellpose_n20`.

## Summary (measured only)

| Metric | threshold | cellpose (same n=20) |
|--------|----------:|---------------------:|
| Dice | **0.054** | **0.930** |
| IoU | 0.028 | 0.869 |
| Instance F1 | 0.435 | 0.904 |
| Mean \|count err\| | 87.5 | 25.7 |
| Rel. count err. | 0.329 | 0.105 |

## Failure mode (honest)

- Otsu on low-contrast phase-contrast **does not recover cell area** (pixel recall ≈ 0.028).
- **4 / 20** FOVs returned **zero objects** (truth 168–425 cells): indices 10, 13, 19, 20 — `SPARSE_FG;ZERO_OBJECTS`.
- Counts can look “plausible” on some frames while masks barely overlap GT (high pixel precision, near-zero recall).

**Product implication:** on dense phase-contrast LIVECell, default to **Cellpose** (or another learned model). Threshold remains useful for fluorescent nuclei (BBBC039), not for this modality.

## Reproduce

```bash
python scripts/run_livecell_validation.py --max-images 20 --split val --backend threshold --skip-download
```
