#!/bin/bash
#SBATCH --job-name=opticell-bbbc039-thr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --output=scripts/ibex/logs/threshold_%j.out
#SBATCH --error=scripts/ibex/logs/threshold_%j.err

# Classical Otsu baseline — full BBBC039 (200 FOVs). No GPU needed.

set -euo pipefail

mkdir -p scripts/ibex/logs outputs/bbbc039_validation data/bbbc039

# Project root = directory containing this repo (submit from repo root)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
elif [[ -f "$HOME/.venv/bin/activate" ]]; then
  source "$HOME/.venv/bin/activate"
fi

echo "Host: $(hostname)"
echo "Date: $(date -Is)"
echo "Python: $(which python)"
python -V

# Skip download if data already present
SKIP=""
if [[ -d data/bbbc039/images && -d data/bbbc039/masks ]]; then
  SKIP="--skip-download"
fi

python scripts/run_bbbc039_validation.py \
  --max-images 200 \
  --backend threshold \
  --out-dir outputs/bbbc039_validation \
  $SKIP

echo "Done: $(date -Is)"
