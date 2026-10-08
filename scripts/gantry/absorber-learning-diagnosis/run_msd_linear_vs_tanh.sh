#!/bin/bash
#SBATCH -J msd-lin-tanh
#SBATCH -p oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=8gb
#SBATCH -t 24:00:00
#SBATCH --array=1-6
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/absorber-learning-diagnosis/msd_linear_vs_tanh_%A_%a.out

# Jan's 3-DOF MSD dynamic parallel, linear vs tanh, 3 seeds each (DECISIONS.md Q2). Does the added
# (5.79 Hz, zeta 0.091) mode get learned, and how reliably? Script: q_msd_linear_vs_tanh.py (a copy
# of Jan's build; no pipeline file is used or changed). CPU only: Jan's Parameterized_MSD_State_Block
# is not GPU-safe. Measured locally: 17 s per epoch, 3000 epochs = about 14 h.
#
# Submit from the repo root on the cluster:
#   mkdir -p /home/dirk_van_den_berg/logs/augmentation/absorber-learning-diagnosis
#   sbatch scripts/gantry/absorber-learning-diagnosis/run_msd_linear_vs_tanh.sh
# Read: grep "epoch" in each log; "3rd pair at X % of points, median f Hz zeta" per 250 epochs.
#
# Task -> run: 1-3 linear seeds 1-3, 4-6 tanh seeds 1-3.

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

T=${SLURM_ARRAY_TASK_ID:?submit as an array}
if [ "$T" -le 3 ]; then KIND=linear; SEED=$T; else KIND=tanh; SEED=$((T - 3)); fi
echo "task=${T} kind=${KIND} seed=${SEED} job=${SLURM_ARRAY_JOB_ID}_${T} node=${SLURM_JOB_NODELIST}"
git rev-parse --short HEAD || true
date

# epochs 3000 (Jan's), chunk 250 (pole check every 250 epochs), SNR 20, approximate baseline
srun --cpu-bind=cores python -u scripts/gantry/absorber-learning-diagnosis/q_msd_linear_vs_tanh.py \
    "$KIND" "$SEED" 3000 250 20 approximate

date
echo "done"
