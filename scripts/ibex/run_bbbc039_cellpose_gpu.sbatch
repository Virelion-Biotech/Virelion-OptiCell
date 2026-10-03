#!/bin/bash
#SBATCH --job-name=opticell-bbbc039-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:1
#SBATCH --output=scripts/ibex/logs/cellpose_%j.out
#SBATCH --error=scripts/ibex/logs/cellpose_%j.err

# Cellpose-SAM (cpsam) on one V100. First run downloads ~1.15G weights.
# Change MAX_IMAGES below (50 pilot → 200 full). Change GPU type if needed:
#   --gres=gpu:rtx2080ti:1   or   --gres=gpu:p100:1

set -euo pipefail

MAX_IMAGES="${MAX_IMAGES:-50}"
CELLPOSE_MODEL="${CELLPOSE_MODEL:-cpsam}"

mkdir -p scripts/ibex/logs outputs/bbbc039_validation data/bbbc039

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
elif [[ -f "$HOME/.venv/bin/activate" ]]; then
  source "$HOME/.venv/bin/activate"
fi

echo "Host: $(hostname)"
echo "Date: $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

SKIP=""
if [[ -d data/bbbc039/images && -d data/bbbc039/masks ]]; then
  SKIP="--skip-download"
fi

python scripts/run_bbbc039_validation.py \
  --max-images "${MAX_IMAGES}" \
  --backend cellpose \
  --cellpose-model "${CELLPOSE_MODEL}" \
  --gpu \
  --out-dir outputs/bbbc039_validation \
  $SKIP

echo "Done: $(date -Is)"
