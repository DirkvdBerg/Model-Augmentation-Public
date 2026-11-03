#!/bin/bash
#SBATCH -J black-box
#SBATCH -p molokai,lanai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH -t 24:00:00
#SBATCH --array=0-10
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/clean/bb_grid_%A_%a.out

# Black-box grid: one array index trains one configuration of GRID on the CPU, since deepSI's eager loop is
# dispatch-bound (about 1.1 h per 20000 updates); then python -m blackbox.select_bb on the job's run folders.
#   sbatch Clean/blackbox/run_black_box.sh                      # all 11
#   sbatch --array=9 Clean/blackbox/run_black_box.sh            # the selected configuration only

set -eo pipefail
IDX=${SLURM_ARRAY_TASK_ID:?run as an array job}
N_ITS=80000   # Adam updates per configuration

# name and arguments per array index: 0 to 8 nx x width at lr 3e-3, 9 and 10 the learning rate at nx 16, width 16
GRID=(
  "nx08_w08        --nx 8  --fh-width 8  --e-width 16 --lr 3e-3"
  "nx08_w16        --nx 8  --fh-width 16 --e-width 32 --lr 3e-3"
  "nx08_w32        --nx 8  --fh-width 32 --e-width 64 --lr 3e-3"
  "nx16_w08        --nx 16 --fh-width 8  --e-width 16 --lr 3e-3"
  "nx16_w16        --nx 16 --fh-width 16 --e-width 32 --lr 3e-3"
  "nx16_w32        --nx 16 --fh-width 32 --e-width 64 --lr 3e-3"
  "nx32_w08        --nx 32 --fh-width 8  --e-width 16 --lr 3e-3"
  "nx32_w16        --nx 32 --fh-width 16 --e-width 32 --lr 3e-3"
  "nx32_w32        --nx 32 --fh-width 32 --e-width 64 --lr 3e-3"
  "nx16_w16_lr1e-3 --nx 16 --fh-width 16 --e-width 32 --lr 1e-3"
  "nx16_w16_lr1e-2 --nx 16 --fh-width 16 --e-width 32 --lr 1e-2"
)
read -r NAME ARGS <<< "${GRID[$IDX]:?no grid line for index $IDX}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
# from Clean/, so the scripts run as python -m blackbox.<script>
cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation/Clean
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

# run folder <index>_<name>_n<updates>_<job>, the form select_bb.py parses
OUT=../simulations/gantry_clean/blackbox/${IDX}_${NAME}_n${N_ITS}_${SLURM_ARRAY_JOB_ID}
echo "index=${IDX} name=${NAME} args=${ARGS} git=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
echo "job_id=${SLURM_JOB_ID} node_list=${SLURM_JOB_NODELIST}"; date
srun --cpu-bind=cores python -u -m blackbox.train_bb $ARGS --n-its "$N_ITS" --out "$OUT"
date
