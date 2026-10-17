#!/bin/bash
#SBATCH -J bb-search
#SBATCH -p molokai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH -t 24:00:00
#SBATCH --array=0-17
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/thesis-results-3/bb_search_%A_%a.out

# Black-box hyperparameter search, batch 1 (DECISIONS.md BB2-029). One array index = one line of
# runners/search_grid.conf (same index). NOT SUBMITTED by the session that wrote it.
#
#   sbatch Thesis-writeup/Code/Server/blackbox/run_bb_search.sh            # all 18
#   sbatch --array=4 Thesis-writeup/Code/Server/blackbox/run_bb_search.sh  # the centre only
#   sbatch --export=ALL,BB_VERSION=lowpass .../run_bb_search.sh   # another data version
#     (noisy = default, noisefree, lowpass = Coulomb-tanh-and-MSD-lowpass-noise)
#
# Thesis R8 (THESIS-RESULTS.md section 4; grid, budget, plateau criterion and limit stated in the
# grid file's header): BB_GRID selects another grid file in runners/, BB_NITS the update budget.
#   sbatch --array=0-10 --export=ALL,BB_VERSION=lowpass,BB_GRID=thesis_r8_grid.conf .../run_bb_search.sh
#   sbatch --array=<selected> -t 48:00:00 #     --export=ALL,BB_VERSION=lowpass,BB_GRID=thesis_r8_grid.conf,BB_NITS=80000 .../run_bb_search.sh
# Final R8 (D-264): the whole grid at the limit with the grey box's objective, no separate retrain:
#   sbatch -p molokai,lanai --array=0-10 --export=ALL,BB_VERSION=lowpass,BB_GRID=thesis_r8_grid.conf,BB_NITS=80000,BB_SHARED_EXTRA="--na 30 --objective nrmse" .../run_bb_search.sh
#
# CPU by default: deepSI's fit is eager and steps the 400-step window one sample at a time through
# small nets, so an update is dispatch-bound, not FLOP-bound (local: batch 256 cost only ~15 % more
# per update than batch 64). A GPU is unlikely to help; to try it on one index:
#   sbatch -p oahu --gres=gpu:1 --array=4 --export=ALL,BB_CUDA=1 .../run_bb_search.sh
# (deepSI moves the model to the GPU for training only; validation free runs stay on the CPU.)
#
# Budget: 20000 Adam updates per configuration, validation every 1000 (20 free runs over the 6
# validation records), deepSI keeps the best validated model. Local estimate (4 threads, batch 256):
# about 4 h at nf 0.1 s, about 5 h for the 32-wide nets, 12 to 16 h at nf 0.4 s (blade3: 2.5 h at
# nf 0.1 s). No --timeout here: with one, deepSI ignores n_its and trains until the time is up
# (job 88256).
# Grep the log for "[B]" (data line, final per-axis result) and "Val sim-RMS" (curve).

set -eo pipefail
IDX=${SLURM_ARRAY_TASK_ID:?run as an array job}
BB_CUDA=${BB_CUDA:-0}
BB_VERSION=${BB_VERSION:-noisy}
BB_GRID=${BB_GRID:-search_grid.conf}
BB_NITS=${BB_NITS:-20000}
# Arguments added to every index of the array (default none). Final R8 (D-264, FIX-18):
#   BB_SHARED_EXTRA="--na 30 --objective nrmse" BB_NITS=80000
BB_SHARED_EXTRA=${BB_SHARED_EXTRA:-}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

D=scripts/gantry/blackbox-cl-trainable
GRID="Thesis-writeup/Code/Server/blackbox/${BB_GRID}"; [ -f "$GRID" ] || GRID="$D/runners/${BB_GRID}"   # thesis grid here, earlier searches in runners/
LINE=$(grep -E "^${IDX}[[:space:]]" "$GRID" || true)
[ -n "$LINE" ] || { echo "no grid line for index ${IDX}"; exit 1; }
NAME=$(echo "$LINE" | awk '{print $2}')
EXTRA=$(echo "$LINE" | awk '{$1=""; $2=""; print}')
CUDA_FLAG=""; [ "$BB_CUDA" = "1" ] && CUDA_FLAG="--cuda"

SHARED="--version ${BB_VERSION} --seed 1 --target e --in64 --f-blockmean --batch 256 --stride 25 \
--n-its ${BB_NITS} --its-per-val 1000 --epochs 1000 ${BB_SHARED_EXTRA}"
TAG=""; [ "$BB_GRID" != search_grid.conf ] && TAG="${BB_GRID%.conf}/"
OUT=$D/outputs/search/${BB_VERSION}/${TAG}${IDX}_${NAME}_n${BB_NITS}_${SLURM_ARRAY_JOB_ID}

echo "index=${IDX} name=${NAME} version=${BB_VERSION} grid=${BB_GRID} n_its=${BB_NITS} extra=${EXTRA} shared_extra=${BB_SHARED_EXTRA} cuda=${BB_CUDA}"
echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
echo "job_id=${SLURM_JOB_ID} array_job=${SLURM_ARRAY_JOB_ID} node_list=${SLURM_JOB_NODELIST}"; date
lscpu | grep -E "Model name" || true
srun --cpu-bind=cores python -u $D/train_bb_clmap.py --server $SHARED $EXTRA $CUDA_FLAG --out "$OUT"
date; echo "done index ${IDX} ${NAME}"
