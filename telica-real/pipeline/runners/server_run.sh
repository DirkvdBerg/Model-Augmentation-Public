#!/bin/bash
#SBATCH -J telica-aug
#SBATCH -p molokai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/telica-real/telica_aug%j.out

# Augmentation training on the REAL Telica gantry (BHL), telica-real/ setup. NOT SUBMITTED by the
# session that wrote it (handoff 2026-09-22): the user launches it.
#
#     sbatch telica-real/pipeline/runners/server_run.sh
#     NX_ANN=8 NF_SECONDS=0.1 sbatch telica-real/pipeline/runners/server_run.sh    # second arm
#
# Every setting comes from telica-real/learnability/ann_settings.json (G5, TR-017/018/020) and
# pipeline/telica_augment.py; the env overrides below are the ONLY knobs here and are echoed into
# the log and into augment_metrics.json.
#
# WHAT THE REPO NEEDS ON THE SERVER: the repo root with telica-real/ (this folder, self-contained:
# vendored code, params, controller/telica_zpk_bhl.mat + telica_sos_identified.npz, baseline/
# recovered_params_a2.json, pipeline/norm_frozen.json, learnability/ann_settings.json) and
# kamtin-data/ (Telica 1.mat for Kt, and "Data Telica/06 40 mm XL 80 mm YL/" for the logs).
# Nothing else of the repo is imported: tr_env.check_no_leak() refuses outside imports.
#
# MEMORY ESTIMATE (TR-020, measured on the dev PC, float64): 80.3 KB per window-step of retained
# graph. Defaults nf = 800 (40 ms), batch 256, checkpoint_chunk 200:
#     live graph   256 x 200 x 80.3 KB  = ~4.1 GB   (unchunked: 256 x 800 x 80.3 KB = ~16.4 GB)
#     host arrays  ~7e4 windows x 800 x 6 x 8 B = ~2.7 GB (stride 20, 117 train records)
#     overhead     ~1-2 GB
# so a 40 GB A100 has a wide margin; on a 16 GB card keep CHUNK <= 200. BATCH / CHUNK scale the
# first line linearly.
#
# RUNTIME: not measured on a GPU (the dev PC has no inductor toolchain). Reference point from the
# simulation pipeline: compiled reduce-overhead, float32, nf 400, batch 512 on the A100 ran
# 1.24 s/update after a ~1 h one-off CUDA-graph recording (NOT a hang). This run is float64 and
# nf 800, so expect 2-4x that per update. 50 epochs x ~270 updates/epoch = ~13.5k updates.
# If compilation fails on this model, rerun with COMPILE_MODE=none (eager, slower, same result).
#
# Sanity lines to grep in the log:
#   "[augment] smoke=False nf=800"                    horizon and settings
#   "[closed loop] Telica controller (identified)"   the identified controller, one row, nc=24
#   "diag(Dc) physical [2.0764e+08 2.7452e+08 3.5731e+08]"   units check of the bank
#   "[real_data] train 117, val 4, test 6"            the split
#   "training rollout COMPILED"                       compilation engaged (unless COMPILE_MODE=none)
#   "[eval] validation: median per-axis RMS ratio"    augmented / baseline-alone, OUTPUT only
#   "[augment] done:"                                 metrics written

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

CACHE_KEY="$(. /etc/os-release; echo "${ID}${VERSION_ID}")-$(uname -m)"
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache/$CACHE_KEY
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache/$CACHE_KEY
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Optional arm overrides (TR-020); empty = the ann_settings.json value.
export NX_ANN="${NX_ANN:-}" NF_SECONDS="${NF_SECONDS:-}" BATCH="${BATCH:-}" CHUNK="${CHUNK:-}"
export EPOCHS="${EPOCHS:-}" LR="${LR:-}" STRIDE="${STRIDE:-}" ITS_PER_VAL="${ITS_PER_VAL:-}"
export COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
unset TELICA_SMOKE

echo "job_id=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST} cache_key=${CACHE_KEY}"
echo "overrides: NX_ANN=${NX_ANN} NF_SECONDS=${NF_SECONDS} BATCH=${BATCH} CHUNK=${CHUNK} EPOCHS=${EPOCHS} LR=${LR} STRIDE=${STRIDE} ITS_PER_VAL=${ITS_PER_VAL} COMPILE_MODE=${COMPILE_MODE}"
git rev-parse --short HEAD || true
nvidia-smi --query-gpu=name,memory.total --format=csv || true
date

# The frozen normalisation must exist (it is committed with telica-real/); never rebuilt here.
test -f telica-real/pipeline/norm_frozen.json

python -u telica-real/pipeline/telica_augment.py

date
