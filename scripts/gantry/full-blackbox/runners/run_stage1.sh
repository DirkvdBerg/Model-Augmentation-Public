#!/bin/bash
# Stage 1 of the full black box: a full ANN (encoder, f and h all MLPs) trained WITHOUT the
# baseline, on the same gantry data and the same I/O normalisation as the augmented run.
# Decides whether the augmented model's "training is net destructive" failure lives in the
# ANN/data/loss/encoder or in the augmentation coupling. Hypothesis, gates and branch are in the
# run table: docs/gantry-augmentation-problem-log.md Section 12 (row STAGE 1 FULL BLACK BOX).
#
# Single job, no array: one model, one lr (1e-3, chosen by scripts/gantry/full-blackbox/lr_probe.py).
# 500 epochs x ~260 Adam updates = ~130k updates. Locally measured 3.6 min/epoch -> ~30 h, so -t is
# set to 36 h for margin. CHECK THIS AGAINST THE PARTITION LIMIT before submitting: the other
# runners in this repo use 8 to 10 h, and if the cap is lower than the estimate the job dies mid-fit.
# On a kill, deepSI's _best/_last checkpoints survive but the npz and the printed gates do NOT,
# because they are written after fit() returns. Lower EPOCHS in blackbox_standalone.py if needed.
#
# No --signal=USR1: the python script installs no USR1 handler, and Python's default action for
# SIGUSR1 is to terminate, which would kill the run early rather than let it checkpoint.
#
# Submit:  sbatch scripts/gantry/full-blackbox/runners/run_stage1.sh
# Watch:   tail -f ~/logs/augmentation/full-blackbox/stage1_<JOBID>.out
# Results: scripts/gantry/full-blackbox/results/<JOBID>/   (config.json, npz, saved system)
#SBATCH -J bb-stage1
#SBATCH -p kauai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=32gb
#SBATCH -t 36:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/full-blackbox/stage1_%j.out

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$(pwd):${PYTHONPATH}"        # repo root -> model_augmentation importable

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p /home/dirk_van_den_berg/logs/augmentation/full-blackbox

echo "========================================"
echo "Stage 1 full black box (full ANN, no baseline)  job=${SLURM_JOB_ID}"
echo "nx=8, na=nb=17, nf=400, lr=1e-3, 500 epochs, val every 5 epochs"
echo "========================================"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
date
echo ""

# Config check, so the log records what actually ran rather than what was intended
# (deployed copies lag local edits).
grep -n "^EPOCHS\|^LR \|^ITS_PER_VAL" scripts/gantry/full-blackbox/blackbox_standalone.py
echo ""

srun --cpu-bind=cores python -u scripts/gantry/full-blackbox/blackbox_standalone.py

echo ""
echo "========================================"
echo "Finished at $(date)"
echo "========================================"
