# OptiCell notebooks

## Colab

| Notebook | Purpose |
|----------|---------|
| [BBBC039_Colab_Validation.ipynb](BBBC039_Colab_Validation.ipynb) | Stage 1 segmentation metrics |
| [Stage2_CTC_Timelapse_Colab.ipynb](Stage2_CTC_Timelapse_Colab.ipynb) | Stage 2 QC\u2192track on CTC TL (**one-by-one**) |

### Stage-2 CTC notebook

```text
https://colab.research.google.com/github/Virelion-Biotech/Virelion-OptiCell/blob/main/notebooks/Stage2_CTC_Timelapse_Colab.ipynb
```

**Always run the install cell first** (`git reset --hard origin/main` + `pip install -e .`). That picks up the tracking LAP fix (no more `cost matrix is infeasible`).

Order:

1. Fluo-N2DH-GOWT1 `01` / `02`
2. Fluo-N2DH-SIM+ `01` / `02`
3. Fluo-N2DL-HeLa `01` / `02`

Defaults: `--backend threshold`, `--enable-tracking`, `track_max_distance=50`, `track_max_gap=1`.

Aggregate: `stage2_ctc_aggregate.csv` \u2014 paste measured rows only.
