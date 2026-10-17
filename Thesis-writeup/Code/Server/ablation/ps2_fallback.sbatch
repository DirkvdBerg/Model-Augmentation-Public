#!/bin/bash
#SBATCH -J ps2-ablation
#SBATCH -p hawaii,oahu,molokai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 03:00:00
#SBATCH --array=0-12
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/ps2-ablation/%x_%A_%a.out

# PS2 fallback ablation (D-270, scripts/gantry/thesis-results/ablate_ps2_fallback.py).
# Submit directly: mkdir -p ~/logs/augmentation/ps2-ablation && sbatch Thesis-writeup/Code/Server/ablation/ps2_fallback.sbatch
# Task 0-9: arms A-E x seeds 1, 2 (task = 2 * arm index + seed - 1); task 10: LSQ (seed 1);
# tasks 11, 12: arm F seeds 1, 2 (D-270 amendment; submit with --array=11-12).
# Expected per task: about 33 min compile, about 11 min for 375 updates (R4 log, oahu); LSQ has no compile.
#
# Read in the output:
#   "[ablation] arm ..."                  the arm and its resolved configuration
#   "[nf val  ]" (first one)              encoder start quality at update 0 (R4 final 1.91e-4 m, run 204 1.89e-5 m)
#   "[ps2] update 150: ..." / "fallback"  the screen (pass <= 0.90)
#   "[ps2] S2 state-map fit: ..."         only when the screen passed
#   "[lsq] ..."                           task 10: per-state MSE, least squares vs Adam 20 / 50 / 100 / 200 epochs

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

ARMS=(A B C D E)
if [ "$SLURM_ARRAY_TASK_ID" -eq 10 ]; then
    export ABL_ARM=LSQ THESIS_SEED=1
elif [ "$SLURM_ARRAY_TASK_ID" -ge 11 ]; then
    export ABL_ARM=F THESIS_SEED=$((SLURM_ARRAY_TASK_ID - 10))
else
    export ABL_ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 2))]} THESIS_SEED=$((SLURM_ARRAY_TASK_ID % 2 + 1))
fi
echo "task=${SLURM_ARRAY_TASK_ID} arm=${ABL_ARM} seed=${THESIS_SEED}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Same inductor/triton caches as the thesis runs (run_thesis.sbatch), keyed by OS image.
CACHE_KEY="$(. /etc/os-release; echo "${ID}${VERSION_ID}")-$(uname -m)"
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache/$CACHE_KEY
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache/$CACHE_KEY
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "job_id=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} node_list=${SLURM_JOB_NODELIST} partition=${SLURM_JOB_PARTITION}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
date

srun --cpu-bind=cores python -u scripts/gantry/thesis-results/ablate_ps2_fallback.py

date
echo "done"
