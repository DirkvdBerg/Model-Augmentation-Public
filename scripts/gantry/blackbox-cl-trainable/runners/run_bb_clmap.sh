#!/bin/bash
#SBATCH -J bb-clmap
#SBATCH -p molokai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/blackbox-cl-trainable/bb_clmap_%j.out

# ONE black-box run, option B on the servo error (DECISIONS.md BB2-018, BB2-019 to BB2-026):
# Jan's SUBNET on the closed-loop map (r, dr, f) -> e = y - r, one seed, one data version, the
# final local configuration with a larger update budget. NOT SUBMITTED by the session that wrote it.
#
#   BB_VERSION=noisy     BB_SEED=1 sbatch scripts/gantry/blackbox-cl-trainable/runners/run_bb_clmap.sh
#   BB_VERSION=noisefree BB_SEED=1 sbatch scripts/gantry/blackbox-cl-trainable/runners/run_bb_clmap.sh
# Optional: BB_LR (default 3e-3: the local proof's 1e-2 diverged on 2 of 8 local runs, e6 and
# noisy seed 3, which deepSI survives only by keeping the best validated model),
# BB_EPOCHS (default 400 = 53k updates), BB_TIMEOUT (default 79200 s = 22 h of training,
# deepSI's own stop, so result.json is written inside the 24 h wall), BB_EXTRA (more flags).
# CPU only: deepSI's fit is sequential over the 400 window steps with tiny nets; no GPU requested.
# Partial result: deepSI keeps the best validated model (validation every 399 updates) and writes
# its _best / _last checkpoints under $OUT/localappdata, so a killed job still leaves a model.
# Grep the log for "[B]" (data line and final per-axis result).

set -eo pipefail
: "${BB_VERSION:?set BB_VERSION=noisy|noisefree}"
: "${BB_SEED:?set BB_SEED=<int>=1}"
BB_LR=${BB_LR:-3e-3}; BB_EPOCHS=${BB_EPOCHS:-400}; BB_TIMEOUT=${BB_TIMEOUT:-79200}; BB_EXTRA=${BB_EXTRA:-}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "BB_VERSION=${BB_VERSION} BB_SEED=${BB_SEED} BB_LR=${BB_LR} BB_EPOCHS=${BB_EPOCHS} BB_TIMEOUT=${BB_TIMEOUT} BB_EXTRA=${BB_EXTRA}"
echo "job_id=${SLURM_JOB_ID} node_list=${SLURM_JOB_NODELIST}"; date
OUT=scripts/gantry/blackbox-cl-trainable/outputs/server/clmap_e_${BB_VERSION}_s${BB_SEED}_${SLURM_JOB_ID}
srun --cpu-bind=cores python -u scripts/gantry/blackbox-cl-trainable/train_bb_clmap.py --server \
    --version "$BB_VERSION" --seed "$BB_SEED" --epochs "$BB_EPOCHS" --timeout "$BB_TIMEOUT" \
    --target e --in64 --dr --lr "$BB_LR" --batch 256 --stride 25 --its-per-val 399 $BB_EXTRA \
    --out "$OUT"
date; echo "done ${BB_VERSION} seed ${BB_SEED}"
