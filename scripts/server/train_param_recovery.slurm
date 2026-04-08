#!/bin/bash
# Example Slurm job for lpv_lfr_baseline parameter recovery.
# Update the partition and conda path to match the target server.

#SBATCH -J lfr-param-recovery
#SBATCH -p hawaii
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -o train_param_recovery_%j.out

set -euo pipefail

# Adjust this if Miniconda/Anaconda is installed elsewhere on the server.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "$REPO_ROOT"

python -m lpv_lfr_baseline.scripts.train_param_recovery
