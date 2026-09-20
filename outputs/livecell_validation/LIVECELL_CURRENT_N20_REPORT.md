# LIVECell current implementation validation — n=20

Measured on the validation split using the same 20 FOVs for every backend.
LIVECell images/annotations are CC BY-NC 4.0; this repository stores metrics, not the dataset.

| Backend | Dice | Instance F1 | Mean \|count error\| |
|---|---:|---:|---:|
| Cellpose-SAM (GPU) | 0.930 | 0.904 | 25.7 |
| Hybrid (GPU) | 0.930 | 0.904 | 25.7 |
| Auto (GPU) | 0.930 | 0.904 | 25.7 |
| Threshold (CPU) | 0.054 | 0.435 | 87.5 |

## Environment

- Google Colab
- NVIDIA Tesla T4
- CUDA-enabled PyTorch
- Cellpose-SAM `cpsam`
- LIVECell validation split
- n=20 FOVs

## Interpretation

- `auto` records the final phase/low-contrast-aware routing behavior. In this n=20 slice, all 20 FOVs resolved to Cellpose.
- `hybrid` records the current count-gated hybrid behavior after the low-contrast override.
- The historical 0.622 hybrid Dice result was measured before the final low-contrast routing rule and is retained only as a pre-rule reference.
- These are measurements on one n=20 validation slice, not claims of universal segmentation performance.
