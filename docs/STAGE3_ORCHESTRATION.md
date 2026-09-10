# Stage 3 — Multi-backend orchestration

**Intent:** avoid silent failures (e.g. SIM+02 threshold TRA=0) by running multiple segmenters and choosing with **auditable rules**, then reporting measured TRA.

## Components

| Piece | Path |
|-------|------|
| Orchestrator CLI | `scripts/run_stage3_orchestrate.py` |
| Colab notebook | `notebooks/Stage3_Orchestration_TRA_Colab.ipynb` |
| CTC export | `scripts/export_ctc_res.py` |

## Selection rule (explicit)

```
rank backends by:
  1. higher mean FOV confidence
  2. lower object-count CV across frames
  3. fewer FOVs with confidence < 50
  4. more frames processed
```

Not a learned model. Failures remain in the table.

## Colab

```text
https://colab.research.google.com/github/Virelion-Biotech/Virelion-OptiCell/blob/main/notebooks/Stage3_Orchestration_TRA_Colab.ipynb
```

Use **GPU**. Mount Drive. Run sequences one-by-one (3 backends \u00d7 TRA each).

## Outputs to commit

- `stage3_tra_by_backend.csv`
- `stage3_autoselect_vs_best.csv`

Only measured TRA/DET/LNK.
