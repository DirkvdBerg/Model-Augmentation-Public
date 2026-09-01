#!/bin/bash
#SBATCH -J gantry-wiring-e2e
#SBATCH -p oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH -t 01:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/augmentation-closed-loop/wiring_e2e%j.out

# D-169. The last gate before a real nf=12000 launch: does compilation survive a REAL fit() call?
# Everything measured so far calls closed_loop_rollout directly from a benchmark.
#
# This also verifies the fit() change. Job 80695 measured the first training update after each
# per-validation CPU/CUDA flip at 599.56 s and 795.69 s, against a 3.30 s eager baseline -- every
# flip forces a Dynamo recompile because .cpu()/.cuda() REPLACE the parameter tensors. fit() no
# longer flips when a simulator is attached; this run is where that shows up or does not.
#
# Three things to grep for in the output:
#   1. NO "skipping cudagraphs due to cpu device"   -> reduce-overhead kept its replay fast path
#   2. NO "[compile] update took ... RECOMPILE"      -> the flip cost is gone
#   3. "[closed loop] training rollout COMPILED"     -> compilation actually engaged

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

# nf=400 (the CURRENT production horizon) keeps the two validations cheap while exercising the
# identical code path. The horizon is not what is under test here; the wiring is.
export E2E_DEVICE=${E2E_DEVICE:-cuda}
export E2E_COMPILE=${E2E_COMPILE:-reduce-overhead}
export E2E_F64=${E2E_F64:-1}
export E2E_NF=${E2E_NF:-400}
export E2E_BATCH=${E2E_BATCH:-512}
export E2E_ITS=${E2E_ITS:-2}

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
date
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "=== end-to-end wiring through a real fit() ==="
srun --cpu-bind=cores python -u scripts/gantry/GPU/wiring_e2e.py

date
echo "done"
