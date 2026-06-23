#!/bin/bash
# Grid: NF_VALUES=[20,40,80,160,200] x LR_VALUES=[5e-4,1e-4,5e-5] = 15 combos
# task_idx -> (nf, lr): [(nf,lr) for nf in NF_VALUES for lr in LR_VALUES]
#   task 0=(20,5e-4)  task 1=(20,1e-4)  task 2=(20,5e-5)
#   task 3=(40,5e-4)  task 4=(40,1e-4)  task 5=(40,5e-5)
#   task 6=(80,5e-4)  task 7=(80,1e-4)  task 8=(80,5e-5)
#   task 9=(160,5e-4) task 10=(160,1e-4) task 11=(160,5e-5)
#   task 12=(200,5e-4) task 13=(200,1e-4) task 14=(200,5e-5)
#
# Submit:  sbatch scripts/gantry/encoder-baseline/runners/run_nf_lr_400hz.sh
# Collect: python scripts/gantry/encoder-baseline/runners/collect_nf_lr_400hz.py
#
# Watch task N live:  tail -f ~/logs/augmentation/encoder/diagnostic_nf_lr_400hz_<JOBID>_N.out
#SBATCH -J diagnostic-nf-lr-400hz
#SBATCH -p kauai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH -t 10:00:00
#SBATCH --array=0-14
#SBATCH --signal=USR1@1800
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/encoder/diagnostic_nf_lr_400hz_%A_%a.out

set -o pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

LOGDIR="/home/dirk_van_den_berg/logs/augmentation/encoder"
mkdir -p "$LOGDIR"
TAG="${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"

echo "========================================"
echo "Diagnostic for NF-LR (job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID})"
echo "NX_ANN=0, nf x lr grid @ 400 Hz"
echo "========================================"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
date
echo ""

python -u scripts/gantry/encoder-baseline/diagnostics/diag_nf_lr_400hz.py \
    --task_idx "${SLURM_ARRAY_TASK_ID}" \
    2>&1 | tee "${LOGDIR}/diagnostic_nf_lr_400hz_${TAG}.log"

echo ""
echo "========================================"
echo "Finished at $(date)"
echo "========================================"
