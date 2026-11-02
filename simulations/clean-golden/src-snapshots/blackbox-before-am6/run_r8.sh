#!/bin/bash
#SBATCH -J bb-r8
#SBATCH -p molokai,lanai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH -t 24:00:00
#SBATCH --array=0-10
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/clean/bb_r8_%A_%a.out

# Black-box grid R8: one array index = one line of thesis_r8_grid.conf, 80000 updates each, on the CPU
# because deepSI's eager loop is dispatch-bound (about 1.1 h per 20000 updates).
# Selection afterwards: Clean/blackbox/select_bb.py on the job's run folders.
#   sbatch Clean/blackbox/run_r8.sh                      # all 11
#   sbatch --array=9 Clean/blackbox/run_r8.sh            # the selected configuration only

set -eo pipefail
IDX=${SLURM_ARRAY_TASK_ID:?run as an array job}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

LINE=$(grep -E "^${IDX}[[:space:]]" Thesis-writeup/Code/Server/blackbox/thesis_r8_grid.conf || true)
[ -n "$LINE" ] || { echo "no grid line for index ${IDX}"; exit 1; }
NAME=$(echo "$LINE" | awk '{print $2}')
EXTRA=$(echo "$LINE" | awk '{$1=""; $2=""; print}')
# <index>_<name>_n<updates>_<job>: the folder name select_bb.py parses
OUT=simulations/gantry_clean/blackbox/${IDX}_${NAME}_n80000_${SLURM_ARRAY_JOB_ID}

echo "index=${IDX} name=${NAME} extra=${EXTRA} git=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
echo "job_id=${SLURM_JOB_ID} node_list=${SLURM_JOB_NODELIST}"; date
srun --cpu-bind=cores python -u Clean/blackbox/train_bb.py $EXTRA --out "$OUT"
date
