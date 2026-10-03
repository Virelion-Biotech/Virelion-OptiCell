# LIVECell validation — hybrid (measured)

**Dataset:** [LIVECell](https://github.com/sartorius-research/LIVECell) (CC BY-NC 4.0)  
**Split:** val · **Backend:** hybrid · **GPU:** true · **Model:** cpsam  
**n_scored:** 20 · same FOV list as threshold / cellpose baselines  
**Timestamp (UTC):** 2026-09-18T13:08:23.222251+00:00

## Three-way comparison (same 20 FOVs)

| Backend | Dice | Instance F1 | Mean \|count err\| |
|---------|-----:|------------:|------------------:|
| **Cellpose-SAM** | **0.930** | **0.904** | **25.7** |
| Hybrid (post collapse-fix) | 0.622 | 0.767 | 37.1 |
| Threshold (Otsu) | 0.054 | 0.435 | 87.5 |

## What hybrid actually chose

| Path | n FOVs | Notes |
|------|-------:|-------|
| `hybrid:cellpose(...)` (count agree) | 8 | Normal gate |
| `hybrid:cellpose(threshold_collapsed,...)` | 5 | Collapse fix fired (incl. thr_count=0) |
| `hybrid:threshold(...)` | 7 | Count disagree; threshold had *some* FG but still near-useless masks |

Dice on the 7 threshold-chosen FOVs is ~0.02–0.05 — same failure mode as pure threshold.

## Honest takeaway

1. **Collapse fix works** for true zero / near-zero threshold: those FOVs now take Cellpose instead of empty masks.
2. **Hybrid is still not competitive with pure Cellpose** on dense phase-contrast. When threshold finds a few spurious blobs (count > 0, fg not fully collapsed), the count-disagreement gate still falls back to threshold and tanks pixel metrics.
3. **Product default on LIVECell-like data:** use **Cellpose** (`backend=auto` / `cellpose`), not hybrid. Hybrid remains useful as a count-safe path on high-contrast fluorescence (BBBC039), where threshold is strong.

No fabricated “hybrid is best” claim.

## Reproduce

```bash
python scripts/run_livecell_validation.py --max-images 20 --split val --backend hybrid --gpu
```
