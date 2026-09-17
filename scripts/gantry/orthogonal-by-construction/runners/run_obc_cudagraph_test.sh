#!/bin/bash
#SBATCH -J obc-cg
#SBATCH -p molokai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 01:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/orthogonal-by-construction/obc_cg%j.out

# ONE QUESTION (now D-199): do fixed-address buffers let CUDA graphs replay instead of re-recording?
#
# Job 83951 measured `reduce-overhead` at 525 s then 1472 s per update at nf 400, degrading, while
# `default` ran at 0.97 s off and 1.89 s on. The hypothesis is that threading thirty freshly
# allocated tensors per objective into the compiled step is what breaks replay, because CUDA
# graphs need stable input addresses.
#
# Three rungs, each in its own process, ~10 min each:
#   1. reduce-overhead --static-pass   the fix
#   2. reduce-overhead                 the control, expected to degrade as in 83951
#   3. reduce-overhead --no-thread     the other control: nothing threaded at all, which is the
#                                      configuration job 83795 runs at 0.77 s
#
# READ: the `steady` figure on the `on` line of each rung, against `default`'s 1.89 s.
# The probe stops a rung after the SECOND production update if it exceeds 60 s, so a degrading
# rung costs one measurement rather than the wall limit.

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
unset OBC_ARM

echo "job_id=${SLURM_JOB_ID}  node=${SLURM_JOB_NODELIST}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || true
date

P=scripts/gantry/orthogonal-by-construction/implementation/08-one-step/probe_compile.py

echo ""; echo "===== 1. reduce-overhead WITH static buffers (the fix) ====="
srun --cpu-bind=cores python -u $P --mode reduce-overhead --static-pass \
    || echo "rung FAILED; continuing"

echo ""; echo "===== 2. reduce-overhead WITHOUT static buffers (control, expect 83951) ====="
srun --cpu-bind=cores python -u $P --mode reduce-overhead \
    || echo "rung FAILED; continuing"

echo ""; echo "===== 3. reduce-overhead, nothing threaded (control, the 83795 configuration) ====="
srun --cpu-bind=cores python -u $P --mode reduce-overhead --no-thread \
    || echo "rung FAILED; continuing"

date; echo done
