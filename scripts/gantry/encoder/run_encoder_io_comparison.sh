#!/bin/bash
#SBATCH -J encoder-io-comparison
#SBATCH -p kauai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH -t 04:00:00
#SBATCH --signal=USR1@1800
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/encoder/encoder_io_comparison_%j.out

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
TAG="${SLURM_JOB_ID:-local}"

echo "========================================"
echo "Encoder I/O comparison  (job $TAG)"
echo "NX_ANN=0 vs NX_ANN=2, nf-step I/O loss"
echo "========================================"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
date
echo ""

python -u scripts/gantry/encoder/encoder_io_comparison.py \
    2>&1 | tee "${LOGDIR}/encoder_io_comparison_${TAG}.log"

echo ""
echo "========================================"
echo "Finished at $(date)"
echo "========================================"
