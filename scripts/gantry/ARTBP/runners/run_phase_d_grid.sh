#!/bin/bash
# Phase D grid (D-120 reframe): ARTBP variance comparison, 4 modes x 5 seeds = 20 tasks.
# task_idx -> (mode, seed), mode-major:  grid = [(m,s) for m in (fixed,geom,poly4,poly6) for s in 0..4]
#   task 0..4   = fixed  seed 0..4        task 10..14 = poly4  seed 0..4
#   task 5..9   = geom   seed 0..4        task 15..19 = poly6  seed 0..4
#
# Submit:  sbatch scripts/gantry/ARTBP/runners/run_phase_d_grid.sh
# Collect: python scripts/gantry/ARTBP/runners/collect_phase_d.py
# Each task writes scripts/gantry/ARTBP/data/train_<mode>_seed<seed>.npz (no collision).
# Watch task N: tail -f ~/logs/augmentation/artbp/phase_d_grid_<JOBID>_N.out
#SBATCH -J artbp-phase-d-grid
#SBATCH -p kauai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=32gb
#SBATCH -t 02:00:00
#SBATCH --array=0-19%5
#SBATCH --signal=USR1@1800
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/artbp/phase_d_grid_%A_%a.out

set -o pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONPATH="$(pwd):${PYTHONPATH}"        # repo root -> model_augmentation importable
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

LOGDIR="/home/dirk_van_den_berg/logs/augmentation/artbp"
mkdir -p "$LOGDIR"
TAG="${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"

echo "========================================"
echo "ARTBP Phase D grid (job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID})"
echo "4 modes x 5 seeds; nf=400, H_max=1600, lr=1e-7, 1 epoch"
echo "node_list=${SLURM_JOB_NODELIST} cpus_per_task=${SLURM_CPUS_PER_TASK}"
date
echo ""

python -u scripts/gantry/ARTBP/train_artbp.py \
    --task_idx "${SLURM_ARRAY_TASK_ID}" \
    2>&1 | tee "${LOGDIR}/phase_d_grid_${TAG}.log"

echo ""
echo "========================================"
echo "Finished at $(date)"
echo "========================================"
