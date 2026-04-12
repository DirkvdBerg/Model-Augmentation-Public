#!/bin/bash
# Single-node 4-GPU smoke test on oahu/blade2.
# Submit from the repo root or any directory with:
#   sbatch scripts/server/run_oahu_blade2_4gpu_test.sh
#
# Make sure this directory exists before submitting:
#   mkdir -p /home/dirk_van_den_berg/logs

#SBATCH -J oahu-4gpu-test
#SBATCH -p oahu
#SBATCH -w blade2
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=16gb
#SBATCH -t 00:15:00
#SBATCH --gres=gpu:4
#SBATCH -o /home/dirk_van_den_berg/logs/oahu_blade2_4gpu_test_%j.out

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "$REPO_ROOT"

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "gpus_requested=4"
echo "pwd=$(pwd)"
echo
echo "=== nvidia-smi -L ==="
nvidia-smi -L
echo
echo "=== nvidia-smi ==="
nvidia-smi
echo
echo "=== python smoke test ==="
python -u scripts/server/test_4gpu_dataparallel.py
