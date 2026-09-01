#!/bin/bash
#SBATCH -J gantry-flip-safety
#SBATCH -p oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH -t 01:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/augmentation-closed-loop/flip_safety%j.out

# D-169, README Sect. 12.2(e). deepSI's fit() moves the model to the CPU before every validation
# and back afterwards (interconnect.py:716,734). With mode='reduce-overhead' the training rollout
# is backed by CUDAGraph Trees holding a GPU memory pool, and flipping devices underneath that is
# untested. Three failure modes, only one of which is loud:
#   1 silent wrong numbers (stale graph pointers)   <- the dangerous one
#   2 a crash                                        <- the good outcome
#   3 silent slowdown (Dynamo recompiles per device, twice an epoch)
#
# This decides between routing validation through the UNCOMPILED hfn (cheap, ~56 min of a 10 h
# run) and compiling it too. It must run BEFORE compilation is wired into train_model: with
# 98 updates per epoch at nf=12000, the first flip with live compiled state lands ~40-45 min
# into a production run, and mode 1 would not announce itself at all.

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

export FLIP_NF=${FLIP_NF:-200}
export FLIP_BATCH=${FLIP_BATCH:-256}
export FLIP_CYCLES=${FLIP_CYCLES:-3}
export FLIP_UPDATES_PER_CYCLE=${FLIP_UPDATES_PER_CYCLE:-5}
export FLIP_VAL_STEPS=${FLIP_VAL_STEPS:-2000}
export FLIP_F64=${FLIP_F64:-1}

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
date
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "=== CPU/CUDA flip safety with compiled training path ==="
srun --cpu-bind=cores python -u scripts/gantry/GPU/flip_safety.py

date
echo "done"
