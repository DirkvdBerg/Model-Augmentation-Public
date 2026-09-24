#!/bin/bash
#SBATCH -J oa-arm
#SBATCH -p molokai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/orthogonal-addition/oa_arm%j.out

# PREPARED, NOT SUBMITTED (handoff 2026-09-23 sect. 2 and G5). One arm of the OBC pair on the
# ORTHOGONAL-ADDITION dataset. Copy of orthogonal-by-construction/runners/run_obc_arm.sh; the only
# changes are the entry file (runners/oa_entry.py, which imports the vendored gantry_dynamic with
# the 14/4/4 record lists) and OA_MODE, the dataset. Every hyperparameter is the production CFG.
#
#     OA_MODE=<dataset> OBC_ARM=obc    sbatch scripts/gantry/orthogonal-addition/runners/run_oa_arm.sh
#     OA_MODE=<dataset> OBC_ARM=noproj sbatch scripts/gantry/orthogonal-addition/runners/run_oa_arm.sh
#
# The dataset folder data/gantry/matlab/trajectory/<OA_MODE>/ must be copied to the cluster first.
# PREDICTION (REPORT.md G5): on this dataset the obc arm's parameters converge to theta* within the
# null-control floor (predicted bias J^+ Delta* inside the G4 band), while noproj does not have that
# guarantee. Submit both from the SAME unedited checkout.
# Sanity lines to grep: "OBC_ARM=", "OA_MODE=", "[obc] reference set:" (671944 tuples from 14
# records), "[obc] refresh at epoch 0" (rank 10 of 10).

set -eo pipefail
if [ -z "${OBC_ARM:-}" ] || [ -z "${OA_MODE:-}" ]; then
    echo "Set both OBC_ARM (noproj | obc) and OA_MODE (the dataset folder name)."; exit 2
fi
case "$OBC_ARM" in noproj|obc) ;; *) echo "OBC_ARM must be noproj or obc"; exit 2 ;; esac
export OBC_ARM OA_MODE

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
CACHE_KEY="$(. /etc/os-release; echo "${ID}${VERSION_ID}")-$(uname -m)"
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache/$CACHE_KEY
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache/$CACHE_KEY
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "OBC_ARM=${OBC_ARM}"
echo "OA_MODE=${OA_MODE}"
echo "job_id=${SLURM_JOB_ID}"
date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
ls "data/gantry/matlab/trajectory/${OA_MODE}/T1_standstill_Ym30.mat" || { echo "dataset not on the cluster"; exit 3; }

srun --cpu-bind=cores python -u scripts/gantry/orthogonal-addition/runners/oa_entry.py
date
echo "done ${OBC_ARM} on ${OA_MODE}"
