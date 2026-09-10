# OptiCell notebooks

## Colab

| Notebook | Purpose |
|----------|---------|
| [BBBC039_Colab_Validation.ipynb](BBBC039_Colab_Validation.ipynb) | Stage 1 segmentation metrics |
| [Stage2_CTC_Timelapse_Colab.ipynb](Stage2_CTC_Timelapse_Colab.ipynb) | Stage 2 QC\u2192track descriptive runs |
| [Stage2_CTC_TRA_Colab.ipynb](Stage2_CTC_TRA_Colab.ipynb) | **TRA / DET / LNK vs CTC GT** (`traccuracy`) |

### TRA notebook

```text
https://colab.research.google.com/github/Virelion-Biotech/Virelion-OptiCell/blob/main/notebooks/Stage2_CTC_TRA_Colab.ipynb
```

Flow per sequence: download CTC \u2192 Stage-2 threshold+tracking \u2192 export RES (`scripts/export_ctc_res.py`) \u2192 `traccuracy` CTCMetrics.

Publish only measured TRA/DET/LNK from `scores/tra_aggregate.csv`.
