# LIVECell validation — Cellpose-SAM (measured)

**Dataset:** [LIVECell](https://github.com/sartorius-research/LIVECell) (CC BY-NC 4.0, non-commercial)  
**Split:** val · **Backend:** cellpose (`cpsam`) · **GPU:** true  
**n_scored:** 20 · **mean / max GT cells per image:** 197.6 / 425  
**Timestamp (UTC):** 2026-09-17T14:56:47.143618+00:00

## Summary (measured only)

| Metric | Value |
|--------|------:|
| Dice | 0.930 |
| IoU | 0.869 |
| Instance F1 | 0.904 |
| Instance precision | 0.944 |
| Instance recall | 0.873 |
| Mean \|count error\| | 25.65 |
| Mean relative count error | 0.105 |

## Honest notes

- Dense phase-contrast (A172 only in this 20-image slice of val filenames).
- Several FOVs flagged `DENSE_FG` (confidence 85); worst count error on the densest image (truth 425, pred 286, |err|=139).
- Pixel Dice stays high (~0.93) even when instance recall drops on crowded frames.
- Threshold baseline not included in this commit (run separately if desired).

## Reproduce

```bash
python scripts/run_livecell_validation.py --max-images 20 --split val --backend cellpose --gpu
```

Artifacts: `livecell_val_cellpose_n20.json`, `livecell_val_cellpose_n20.csv`.
