# LIVECell current implementation validation — n=20

Measured on the validation split using the same first 20 FOVs selected by the validation script.
LIVECell images/annotations are CC BY-NC 4.0; this repository stores metrics, not the dataset.

| Backend | Dice | Instance F1 | Mean count error |
|---|---:|---:|---:|
| cellpose | 0.930 | 0.904 | 25.6 |
| hybrid | 0.930 | 0.904 | 25.6 |
| auto | 0.930 | 0.904 | 25.6 |
| threshold | 0.054 | 0.435 | 87.5 |

## Interpretation

- `auto` records the final phase/low-contrast-aware routing behavior.
- `hybrid` records the current count-gated hybrid behavior after the low-contrast override.
- The historical 0.622 hybrid Dice result was measured before the final low-contrast routing rule and is retained only as a pre-rule reference.
- These are benchmark measurements on one n=20 slice, not claims of universal segmentation performance.
