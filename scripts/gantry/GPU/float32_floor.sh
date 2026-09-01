#!/bin/bash
#SBATCH -J gantry-float32-floor
#SBATCH -p oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH -t 02:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/augmentation-closed-loop/float32_floor%j.out

# D-169. Is a normalised residual of 1e-7/1e-8 an OPTIMISATION problem or an ARITHMETIC one?
#
# Measures |y_float32 - y_float64| for the same rollout at several horizons. That difference is
# float32's uncertainty about its own answer, so no amount of training can produce a residual
# below it. Compare against the residual actually being fitted and the answer falls out.
#
# Context: job 80610 measured eager-vs-compiled at max|dy| = 7.2e-07 at nf=200, which is ~6 x
# float32 eps -- i.e. that disagreement is the ARITHMETIC, not the compiler. This job puts a
# number on the arithmetic itself, at the horizon that matters.
#
# Runs on `oahu` deliberately even though it exercises FP64 (1/32 rate on a 2080 Ti): the float64
# arm here is a REFERENCE for one rollout, not a training proposal, and the floor is a property
# of the numbers rather than of the card. If the result says float64 is NEEDED, the follow-up
# belongs on lanai/molokai (A100, FP64 1/2 rate), and that is a different job.

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

# Small batch: the floor is a per-sample property, not a statistical one, and FP64 on a 2080 Ti
# is 1/32 rate so a wide batch buys nothing but wall time.
export FLOOR_BATCH=${FLOOR_BATCH:-64}
export FLOOR_HORIZONS=${FLOOR_HORIZONS:-200,1000,4000,12000}

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
date
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "=== float32 arithmetic floor vs horizon ==="
srun --cpu-bind=cores python -u scripts/gantry/GPU/float32_floor.py

date
echo "done"
