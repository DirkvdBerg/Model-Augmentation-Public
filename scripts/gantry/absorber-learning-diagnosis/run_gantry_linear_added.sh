#!/bin/bash
#SBATCH -J gantry-linear-added
#SBATCH -p hawaii
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/absorber-learning-diagnosis/gantry_linear_added_%j.out

# Thesis run 42 with LINEAR added-state dynamics (absorber-learning diagnosis, DECISIONS.md S1).
# Script: scripts/gantry/absorber-learning-diagnosis/gantry_linear_added.py, a copy of
# gantry_interconnect_dynamic.py whose only change (marked "LINEAR ADDED") replaces the network
# inside the ANN block: linear state and readout nets for the added states, tanh static net.
# Everything else is the thesis design, set by the THESIS_* variables below exactly as run 42:
#   arm U, noisy data, detuned start, n_a 2, seed 1, nf 0.1 s; float64, batch 512, lr 1e-5,
#   2x16 nets, epochs 200, L-BFGS polish 100 iterations, device cuda, compile reduce-overhead.
#
# Submit from the repo root:
#   mkdir -p /home/dirk_van_den_berg/logs/augmentation/absorber-learning-diagnosis
#   sbatch scripts/gantry/absorber-learning-diagnosis/run_gantry_linear_added.sh
#
# Sanity lines to grep in the output:
#   "[thesis run] arm=u noise=noisy start=detuned n_a=2 seed=1"
#   "[linear added] ann.net replaced"      the change engaged
#   "training rollout COMPILED"            compilation engaged

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# The run: thesis run 42's variables (runs.tsv id 42).
export THESIS_ARM=u THESIS_NOISE=noisy THESIS_START=detuned THESIS_NA=2 THESIS_SEED=1 THESIS_NF=0.1

# Do NOT clear CUDA_VISIBLE_DEVICES here: --gres=gpu:1 sets it to the allocated card, and the
# config refuses a device index for exactly that reason. Clearing it makes device='cuda' fail.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Keep inductor/triton codegen off $HOME (quota) and warm across runs, KEYED BY OS IMAGE (job 85010
# died on blade2 importing a launcher an A100 job had compiled against a newer GLIBC).
CACHE_KEY="$(. /etc/os-release; echo "${ID}${VERSION_ID}")-$(uname -m)"
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache/$CACHE_KEY
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache/$CACHE_KEY
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
echo "cache_key=${CACHE_KEY}"

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
date

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "=== toolchain (inductor needs gcc and triton) ==="
which gcc && gcc --version | head -n 1 || echo "gcc NOT FOUND -> inductor will fail"
python -c "import triton; print('triton', triton.__version__)" || echo "triton NOT FOUND"

echo "=== git (the code this run used) ==="
git rev-parse --short HEAD || true

echo "=== gantry_linear_added (thesis run 42 settings, linear added states) ==="
srun --cpu-bind=cores python -u scripts/gantry/absorber-learning-diagnosis/gantry_linear_added.py

date
echo "done"
