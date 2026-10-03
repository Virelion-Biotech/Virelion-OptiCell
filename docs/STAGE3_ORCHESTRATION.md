# Stage 3 — Multi-backend orchestration

**Intent:** avoid silent failures (e.g. SIM+02 threshold TRA=0) by running multiple segmenters and choosing with **auditable rules**, then reporting measured TRA.

## Selection rule v2

Derived from measured CTC Stage-3 TRA (Cellpose best on 6/6; thr/hybrid collapse when `count_cv > 1`):

1. **Reject** failed runs, zero mean count, `count_cv > 1.0`, or `zero_object_fraction > 0.25`
2. **Prefer** `cellpose` > `hybrid` > `threshold` among survivors
3. **Tie-break** lower `count_cv`, higher mean confidence, more frames

Do **not** rank primarily by FOV confidence when it saturates at 100.

Offline check (no GPU):

```bash
python scripts/validate_stage3_selection.py
# expects Match rate: 6/6 on outputs/stage3/stage3_tra_by_backend.csv
```

## Components

| Piece | Path |
|-------|------|
| Orchestrator CLI | `scripts/run_stage3_orchestrate.py` |
| Selection validator | `scripts/validate_stage3_selection.py` |
| Colab notebook | `notebooks/Stage3_Orchestration_TRA_Colab.ipynb` |
| Report | `outputs/stage3/STAGE3_REPORT.md` |

## Colab

```text
https://colab.research.google.com/github/Virelion-Biotech/Virelion-OptiCell/blob/main/notebooks/Stage3_Orchestration_TRA_Colab.ipynb
```

GPU recommended. Publish only measured TRA tables.
