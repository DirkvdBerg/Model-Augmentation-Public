#!/bin/bash
#SBATCH -J gantry-compile-precision
#SBATCH -p oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH -t 02:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/augmentation-closed-loop/compile_precision%j.out

# D-169 acceptance gate. gpu_bench2 (job 80610) measured inductor+reduce-overhead at 6.06x
# (2.487 ms/step, 1206 updates in a 10 h wall at nf=12000) with max|dg|=7.5e-10 on ONE
# forward/backward. This asks the question that decides whether that speedup is usable: after
# N optimiser steps from identical parameters on identical batches, do eager and compiled end
# up in the same place?
#
# Compared against the float32-vs-float64 result this project already ACCEPTED
# (cos=0.999042, ratio=1.0048), so the verdict is relative to a precedent rather than to an
# arbitrary tolerance.
#
# Default PREC_NF=200 runs in ~5 min and answers the question in principle.
# PREC_NF=12000 is the DEFINITIVE run at the real training horizon (~40-60 min); prefer it
# before committing a 10 h job:
#     PREC_NF=12000 PREC_UPDATES=20 sbatch scripts/gantry/GPU/compile_update_precision.sh

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

export PREC_NF=${PREC_NF:-200}
export PREC_BATCH=${PREC_BATCH:-512}
export PREC_UPDATES=${PREC_UPDATES:-40}
export PREC_CHUNK=${PREC_CHUNK:-0}

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
date
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "=== compile vs eager: update-level precision ==="
srun --cpu-bind=cores python -u scripts/gantry/GPU/compile_update_precision.py

date
echo "done"
