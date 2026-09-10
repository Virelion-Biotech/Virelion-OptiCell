# OptiCell notebooks

## Colab

| Notebook | Purpose |
|----------|---------|
| [BBBC039_Colab_Validation.ipynb](BBBC039_Colab_Validation.ipynb) | Stage 1 segmentation metrics |
| [Stage2_CTC_Timelapse_Colab.ipynb](Stage2_CTC_Timelapse_Colab.ipynb) | Stage 2 QC→track on CTC TL datasets **one-by-one** |

### Open Stage-2 CTC notebook in Colab

1. Open: https://colab.research.google.com/
2. File → Open notebook → GitHub
3. Repo: `Virelion-Biotech/Virelion-OptiCell`
4. Path: `notebooks/Stage2_CTC_Timelapse_Colab.ipynb`

Or direct pattern:

```text
https://colab.research.google.com/github/Virelion-Biotech/Virelion-OptiCell/blob/main/notebooks/Stage2_CTC_Timelapse_Colab.ipynb
```

Runs (in order):

1. Fluo-N2DH-GOWT1 `01` then `02` (~53 MB zip)
2. Fluo-N2DH-SIM+ `01` then `02` (~91 MB)
3. Fluo-N2DL-HeLa `01` then `02` (~182 MB)

Each cell downloads only if needed, then runs `scripts/run_killer_workflow.py --backend threshold --enable-tracking`.

**CPU runtime is enough.** Set `MAX_FRAMES = 20` for a fast pilot; `0` for full sequences.

Measured aggregate table is written to `stage2_ctc_aggregate.csv` — paste that back for the Stage-2 report (no invented TRA scores).
