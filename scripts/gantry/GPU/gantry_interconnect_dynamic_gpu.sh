#!/bin/bash
#SBATCH -J gantry-interconnect-dynamic-gpu
#SBATCH -p oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH --signal=USR1@1800
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/augmentation-closed-loop/gantry_interconnect_dynamic%j.out

# The full training run on the GPU. Same entry point as the CPU script; the device, the
# compilation mode and every hyperparameter come from CFG in gantry_interconnect_dynamic.py,
# NOT from here. This file only supplies the allocation and the environment.
#
# Expected config (verify against the run's own "Configuration:" block, which is authoritative):
#   device=cuda  compile_mode=reduce-overhead  nf=400  batch=512  lr=3e-5
#   nx_ann=8  24x3  na_nb=29  epochs=350  its_per_val=650  stride=10  chunk=0  float32
#
# Measured on blade1 (RTX 2080 Ti), job 80713, at nf=400 batch 512:
#   0.50 s/update steady state, against 5.0 s/update for the CPU sweep 80498-80557.
#   ~500 s of ONE-OFF compile on the first two updates; the cache dirs below make repeat
#   launches cheaper. NOTE 80713 ran the smaller nx_ann=2 16x2 model, so expect somewhat
#   above 0.50 s/update here: the loop is dispatch-bound and 24x3 adds ops per step.
#
# Sanity lines to grep in the output:
#   "training rollout COMPILED"            compilation engaged
#   NO "skipping cudagraphs due to cpu"    CUDA graphs kept their fast path
#   "[nf val ] rms ..."                    the window probe ran (not "failed (non-fatal)")

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1

# Do NOT clear CUDA_VISIBLE_DEVICES here: --gres=gpu:1 sets it to the allocated card, and the
# config refuses a device index for exactly that reason. Clearing it makes device='cuda' fail.

# Still needed with a GPU: data loading, the window build and the eager validation free run are
# all CPU work, and validation is the single largest per-epoch cost once training is compiled.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Keep inductor/triton codegen off $HOME (quota) and warm across runs.
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
date

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "=== host CPU (t_step tracks HOST dispatch, not the GPU) ==="
lscpu | grep -E "CPU\(s\)|Thread|Core|Socket|Model name"

echo "=== toolchain (inductor needs gcc and triton) ==="
which gcc && gcc --version | head -n 1 || echo "gcc NOT FOUND -> inductor will fail"
python -c "import triton; print('triton', triton.__version__)" || echo "triton NOT FOUND"

echo "=== gantry_interconnect_dynamic GPU ==="
srun --cpu-bind=cores python -u scripts/gantry/gantry_interconnect_dynamic.py

date
echo "done"
