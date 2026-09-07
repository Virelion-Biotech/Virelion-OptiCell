# Running OptiCell BBBC039 validation on KAUST Ibex

Do **not** run long jobs on login nodes. Use `sbatch` from `/ibex/user/$USER`.

## 1. One-time setup on Ibex

```bash
# Prefer GPU login if you will use Cellpose
ssh -X $USER@glogin.ibex.kaust.edu.sa
# or CPU login:
# ssh -X $USER@ilogin.ibex.kaust.edu.sa

mkdir -p /ibex/user/$USER/projects
cd /ibex/user/$USER/projects
git clone https://github.com/Virelion-Biotech/Virelion-OptiCell.git
cd Virelion-OptiCell

# Python env (conda or venv — adjust to what Ibex provides)
module avail python   # inspect
module load python/3.11   # example; use what module avail shows

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[cellpose]"
pip install packaging

# Optional: confirm GPU torch sees CUDA on a GPU node later
python -c "import torch; print('cuda', torch.cuda.is_available())"
```

If BBBC039 is already on your laptop, copy it (saves re-download):

```bash
# from your laptop
scp -r ~/Virelion-OptiCell/data/bbbc039 $USER@ilogin.ibex.kaust.edu.sa:/ibex/user/$USER/projects/Virelion-OptiCell/data/
```

Otherwise the job scripts download on first run.

## 2. Submit jobs

```bash
cd /ibex/user/$USER/projects/Virelion-OptiCell
source .venv/bin/activate

# Fast classical baseline (CPU, full 200 FOVs)
sbatch scripts/ibex/run_bbbc039_threshold_cpu.sbatch

# Cellpose-SAM on 1x V100 GPU (50 FOVs first; edit MAX_IMAGES for 200)
sbatch scripts/ibex/run_bbbc039_cellpose_gpu.sbatch

# Status
squeue -u $USER
# Logs land in scripts/ibex/logs/
```

## 3. After jobs finish

```bash
ls -la outputs/bbbc039_validation/
cat outputs/bbbc039_validation/bbbc039_*_n*.json | head
```

Copy measured JSON/CSV back to your laptop and we can commit reports with **only measured numbers**.

## Notes

- Partition default on Ibex is `batch`.
- GPU script requests `--gres=gpu:v100:1`. Change to `rtx2080ti`, `p100`, etc. if queue is full (see SLURM cheat sheet).
- Walltime: threshold ~30 min; Cellpose 50 FOVs on V100 usually under 2 h (model download once).
- Never invent metrics — paste SUMMARY from the `.out` log.
