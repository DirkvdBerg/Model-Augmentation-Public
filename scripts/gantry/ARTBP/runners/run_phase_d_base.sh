#!/bin/bash
# Phase D grid (D-120), SINGLE job (non-array): runs all 4 modes x 5 seeds = 20 combos sequentially
# in one process (~12 min each -> ~4 h total). Each writes scripts/gantry/ARTBP/data/train_<mode>_seed<seed>.npz.
# Submit:  sbatch scripts/gantry/ARTBP/runners/run_phase_d_base.sh
# Collect: python scripts/gantry/ARTBP/runners/collect_phase_d.py
#SBATCH -J artbp-phase-d
#SBATCH -p kauai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=48gb
#SBATCH -t 08:00:00
#SBATCH --signal=USR1@1800
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/artbp/phase_d_base%j.out

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$(pwd):${PYTHONPATH}"        # repo root -> model_augmentation importable

# Force CPU usage.
export CUDA_VISIBLE_DEVICES=""

# Use the CPUs allocated by SLURM.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# All 20 combos in one process (no --task_idx -> uses these env lists).
export TA_MODES="fixed,geom,poly4,poly6"
export TA_SEEDS="0,1,2,3,4"

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "modes=${TA_MODES} seeds=${TA_SEEDS}"
date

echo "=== CPU info ==="
lscpu | grep -E "CPU\(s\)|Thread|Core|Socket|Model name"

echo "=== ARTBP Phase D grid (single job) ==="
srun --cpu-bind=cores python -u scripts/gantry/ARTBP/train_artbp.py

echo ""
echo "Finished at $(date)"
