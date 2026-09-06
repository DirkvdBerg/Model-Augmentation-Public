#!/bin/bash
#SBATCH -J lbfgs-gpu-probe
#SBATCH -p hawaii
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 02:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/lbfgs/lbfgs_gpu_probe%j.out

# RUN THIS ONCE BEFORE THE FIRST SUBMISSION, or the job dies at launch and never appears in
# squeue:
#     mkdir -p ~/logs/augmentation/lbfgs
# SLURM opens the -o path BEFORE this script starts, so a `mkdir` further down cannot help. That
# missing directory is the most likely reason a submitted job leaves no trace anywhere.
#
# PARTITION. `hawaii` on purpose, not `oahu`. Job 81262, the checkpoint this probe measures
# against, ran there ("device=cuda (Quadro RTX 6000)", 24 GB). The memory table this probe
# produces is a property of the CARD, so probe on the partition you will actually run on:
# oahu/mpi are RTX 2080 (8 GB) and lanai/molokai are A100 (40-80 GB), and their ceilings differ
# by a factor of 3 to 10.

# Measurement only. Runs no optimisation, changes no weights, writes no checkpoint.
# Answers the three questions tasks/lbfgs-known-issues.md leaves open (D-171):
#   Q1  is the closure bit-deterministic on CUDA, eager and compiled?
#   Q2  what does one closure cost in time and memory, over windows x chunk x dtype?
#   Q3  does the batch build scale with the request rather than the dataset?
#
# Two hours is generous: each point is one warm-up plus one timed closure, ~40 closures per
# dtype. Most of the wall is data loading and the four model builds.
#
# What to do with the output:
#   * VERDICT block, 'compiled ... loss deterministic=True' -> leave force_eager=False and the
#     phase keeps the ~6.5x compile speedup. False -> it falls back to eager by itself, loudly.
#   * Q2 table -> pick lbfgs_windows as the largest that fits with margin AT THE DTYPE YOU WILL
#     RUN, and set lbfgs_chunk when the single-chunk row OOMs and the chunked one does not.
#   * Q3 'built' vs the full set -> confirms the stride scaling. 'built' should be a small
#     multiple of 'windows', never the full 66612.
#
# REQUIRES on the cluster, and they must be consistent with each other:
#   model_augmentation/fit_systems/lbfgs_polish.py            (new)
#   scripts/gantry/gantry_dynamic/config.py                   (modified)
#   scripts/gantry/gantry_dynamic/training.py                 (modified)
#   scripts/gantry/gantry_interconnect_dynamic.py             (modified)
#   scripts/gantry/gantry_dynamic/tests/test_lbfgs_polish.py  (new, for the unit step below)
#   the checkpoint and the augmentation_ma50_b140-230_a6_z03 dataset
#
# THEN, and only then, the real test: resume the best checkpoint into the phase with
#   RESUME_CHECKPOINT=.../SSE_Interconnect_Composed_p2LPDA_best  and  start_phase='lbfgs'
# The bar is a sim-RMS improvement beyond that run's own +-0.1% validation oscillation, so ~0.3%.

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1

# Do NOT clear CUDA_VISIBLE_DEVICES: --gres=gpu:1 sets it to the allocated card and the config
# refuses a device index for exactly that reason.

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

echo "=== GPU info (the memory table below is a property of THIS card) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "=== toolchain (inductor needs gcc and triton; without them the compiled arm is skipped) ==="
which gcc && gcc --version | head -n 1 || echo "gcc NOT FOUND -> inductor will fail"
python -c "import triton; print('triton', triton.__version__)" || echo "triton NOT FOUND"

# The unit gates first: ~2.5 minutes, mostly device-independent, and a failure here means the
# probe below would be measuring something broken. These run on the CPU by construction (the
# test config sets device='cpu'); the GPU questions are the probe's job.
echo "=== unit gates ==="
srun --cpu-bind=cores python -u -m unittest discover \
    -s scripts/gantry/gantry_dynamic/tests -p "test_lbfgs_polish.py" -v

echo "=== GPU probe ==="
srun --cpu-bind=cores python -u scripts/gantry/lbfgs/lbfgs_gpu_probe.py

date
echo "done"
