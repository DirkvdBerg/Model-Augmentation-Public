#!/bin/bash
#SBATCH -J encoder-start
#SBATCH -p molokai,hawaii,oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 01:00:00
#SBATCH --array=0-5
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/thesis-results-4/%x_%A_%a.out

# T2 of the least-squares encoder start (D-271, tasks/encoder-lsq-proposal-2026-10-17.md): full data, start only,
# no training. Task = problem + 3 * repeat; problem 0 = (detuned, seed 1): R1 R4 R13 R14; 1 = (correct, seed 1):
# R2 R3 R5 R6 R7; 2 = (detuned, seed 2), an independent case. A few minutes per task.
# Submit: mkdir -p ~/logs/augmentation/thesis-results-4 && sbatch Thesis-writeup/Code/Server/greybox/encoder_start_check.sbatch
# Pass: every task ends with "[T2] ... start set, hash ..." (the readiness check R^2 >= 0.999 per state passed), and
# tasks p and p+3 print the same hash. Read: grep -h "\[encoder start\]\|\[T2\]\|Error" <logs>

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
export ENC_T2=$((SLURM_ARRAY_TASK_ID % 3))

echo "task=${SLURM_ARRAY_TASK_ID} problem=${ENC_T2} repeat=$((SLURM_ARRAY_TASK_ID / 3)) node_list=${SLURM_JOB_NODELIST}"
date
srun --cpu-bind=cores python -u scripts/gantry/thesis-results/test_encoder_lsq.py
date
echo "done"
