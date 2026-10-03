# LIVECell T4 validation procedure

The scientific LIVECell benchmark is run on a CUDA GPU such as a Colab Tesla T4. Ordinary GitHub Actions runners do not provide a T4-class GPU, so the full Cellpose n=20 inference benchmark is not run in ordinary CI.

## Clean Colab environment

Use a fresh runtime. Do not upgrade setuptools independently before installing OptiCell.

```bash
%%bash
set -euo pipefail

git clone --depth 1 https://github.com/Virelion-Biotech/Virelion-OptiCell.git /content/Virelion-OptiCell
cd /content/Virelion-OptiCell

python -m pip install -e ".[cellpose]"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA/T4 is required for the GPU benchmark"
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY
```

## Measured benchmark

```bash
%%bash
set -euo pipefail
cd /content/Virelion-OptiCell

python scripts/run_livecell_validation.py   --split val   --max-images 20   --backend hybrid   --gpu

python scripts/run_livecell_validation.py   --split val   --max-images 20   --backend auto   --gpu   --skip-download

python scripts/write_livecell_comparison.py
python scripts/validate_livecell_artifacts.py
```

The committed n=20 record uses the same validation FOVs for Cellpose, hybrid, auto, and threshold. The LIVECell source images and annotations are not committed because they are distributed under CC BY-NC 4.0.

## Environment hygiene

Do not run `pip install --upgrade setuptools` before this benchmark. PyTorch may require `setuptools<82` in the Cellpose environment; letting pip resolve the project dependencies avoids the global-environment conflict observed during validation.
