#!/bin/bash
#SBATCH -J clean-l4
#SBATCH -p kauai,molokai,lanai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=32gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/clean-code/%x_%A_%a.out

# Layers 4 and 4b of the clean-code harness (tasks/clean-code/design.md 6.3, D-275 amendment 4a).
# Array task 0 = mainline (with --eval-check: the loader proven against the unmodified evaluation),
# task 1 = clean. One thread each (bitwise comparison). The staging folder must exist first
# (l4_stage.py). Afterwards: python -u compare_l4.py $OUT/mainline $OUT/clean

set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
REPO=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
cd "$REPO/Thesis-writeup/Code/Verification/clean-code"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
STAGING="${STAGING:-$REPO/simulations/clean-golden/L4-staging}"
OUT="${OUT:-$REPO/simulations/clean-golden/L4}"
SIDES=(mainline clean)
SIDE=${SIDES[${SLURM_ARRAY_TASK_ID:?submit with --array=0-1}]}
EXTRA=""; [ "$SIDE" = mainline ] && EXTRA="--eval-check"
echo "side=${SIDE} staging=${STAGING} out=${OUT}/${SIDE} node=${SLURM_JOB_NODELIST}"; date
python -u l4_loader.py --side "$SIDE" --staging "$STAGING" --out "$OUT/$SIDE" $EXTRA
date; echo "done ${SIDE}"
