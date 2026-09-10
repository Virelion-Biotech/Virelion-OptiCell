# OptiCell notebooks

| Notebook | Stage | Purpose |
|----------|-------|---------|
| [BBBC039_Colab_Validation.ipynb](BBBC039_Colab_Validation.ipynb) | 1 | Segmentation metrics |
| [Stage2_CTC_Timelapse_Colab.ipynb](Stage2_CTC_Timelapse_Colab.ipynb) | 2 | Descriptive TL runs |
| [Stage2_CTC_TRA_Colab.ipynb](Stage2_CTC_TRA_Colab.ipynb) | 2 | TRA vs CTC GT (threshold) |
| [Stage3_Orchestration_TRA_Colab.ipynb](Stage3_Orchestration_TRA_Colab.ipynb) | **3** | **Multi-backend + auto-select + TRA (GPU)** |

### Stage 3 (recommended next run)

```text
https://colab.research.google.com/github/Virelion-Biotech/Virelion-OptiCell/blob/main/notebooks/Stage3_Orchestration_TRA_Colab.ipynb
```

- Runtime: **GPU**
- Backends: threshold, hybrid, cellpose (`cpsam`)
- Data: all CTC training sequences (GOWT1, SIM+, HeLa)
- Outputs: TRA per backend + whether auto-select matched best TRA
