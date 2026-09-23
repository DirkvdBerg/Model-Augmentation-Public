#!/bin/bash
#SBATCH -J telica-real-aug
#SBATCH -p oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/augmentation-closed-loop/telica_real_aug%j.out

# Augmentation training on the REAL Telica gantry (BHL), telica-real/ setup. NOT SUBMITTED by the
# session that wrote it: the user launches it. Conventions follow the user's working GPU runner
# (gantry_interconnect_dynamic GPU script), TR-024.
#
#     sbatch telica-real/pipeline/runners/server_run.sh                 # fixed G4 baseline + ANN
#     NX_ANN=6 sbatch telica-real/pipeline/runners/server_run.sh          # a second arm
#     FS_TRAIN=20000 STRIDE=20 CHUNK=200 sbatch ...                         # the 20 kHz path
#
# THE THREE PROJECTION ARMS (TR-025). Joint estimation of the ten identifiable combinations + cc,
# started at the G4 values, no parameter prior; the arms differ ONLY in the projection. Submit all
# three from the SAME unedited checkout, or the comparison silently stops being paired:
#     OBC_ARM=noproj     sbatch telica-real/pipeline/runners/server_run.sh    # no projection
#     OBC_ARM=obc        sbatch telica-real/pipeline/runners/server_run.sh    # 13-column tangent OBC
#     OBC_ARM=obc_affine sbatch telica-real/pipeline/runners/server_run.sh    # + frozen offset column
# The OBC arms build a reference set from every train sample (~3e5 tuples at 5 kHz) and refresh the
# basis each epoch; measured on the simulation pipeline OBC costs ~4x per update, so they are the
# slow arms. Gates passed locally, no optimizer step (run g6_joint_a2): friction tangent == jvp
# (2e-16), OBC orthogonality 2e-13, basis rank 13/13 at the G4 point.
#
# Every setting comes from telica-real/learnability/ann_settings.json and
# telica-real/pipeline/telica_augment.py; the env overrides below are the ONLY knobs here and are
# echoed into the log and into augment_metrics.json.
#
# Expected config (the run's own "[augment] smoke=False ..." line is authoritative):
#   rate 5 kHz (TR-022)  nf=200 (40 ms)  batch=256  lr=1e-5  nx_ann=8  24x3  na_nb=29
#   route (3,4,5,6..13)  epochs=50  its_per_val=1400  stride=5  chunk=0  float64 (TR-013)
#   device=cuda  compile_mode=reduce-overhead  controller: identified_5k (TR-022)
#
# WHAT THE REPO NEEDS ON THE SERVER: the repo root with telica-real/ (self-contained: vendored
# code, params, controller/telica_zpk_bhl.mat + telica_sos_identified.npz +
# telica_sos_identified_5k.npz, baseline/recovered_params_a2.json, pipeline/norm_frozen.json,
# learnability/ann_settings.json) and kamtin-data/ (Telica 1.mat for Kt, and
# "Data Telica/06 40 mm XL 80 mm YL/" for the logs). Nothing else of the repo is imported:
# tr_env.check_no_leak() refuses outside imports.
#
# MEMORY (TR-020/TR-022/TR-024, measured on the dev PC, float64, nx_ann 8, run g6_probe_5k_nx8):
# 81.7 KB per window-step of retained graph; batch 256 x 200 steps = ~4.2 GB, no checkpointing
# needed. Host window arrays ~0.2 GB. At FS_TRAIN=20000 use STRIDE=20
# CHUNK=200 (800 steps, ~4 GB with chunking).
#
# RUNTIME: not measured on a GPU. Reference from the simulation runner: 0.50 s/update on an
# RTX 2080 Ti (job 80713, nf 400, batch 512, FLOAT32, nx_ann 2 16x2) and 1.24 s/update on the A100
# (OBC, float32, nf 400), each after a ONE-OFF compile (~500 s to ~1 h of CUDA-graph recording,
# NOT a hang). This run is FLOAT64: consumer cards (RTX) run float64 at ~1/32 of float32, the A100
# at 1/2; the loop is dispatch-bound, so the real penalty is unknown until the first updates. Check
# the "GPU info" block below and the first timing lines. If compilation fails, rerun with
# COMPILE_MODE=none (eager, slower, same result).
#
# Sanity lines to grep in the output:
#   "[augment] smoke=False nf=200"                   rate / horizon / nx_ann as expected
#   "[real_data] train 117, val 4, test 6"           the supervisor's split (folders), all read
#   "[closed loop] Telica controller (identified_5k)" the identified controller at 5 kHz, one row
#   "training rollout COMPILED"                      compilation engaged
#   NO "skipping cudagraphs due to cpu"              CUDA graphs kept their fast path
#   "[nf val  ] rms ..."                             the window probe ran (not "failed (non-fatal)")
#   "[eval] validation: median per-axis RMS ratio"   augmented / baseline-alone, OUTPUT only
#   "[augment] done:"                                metrics written
# With OBC_ARM set, also:
#   "[augment] arm=obc joint_estimation=True obc=True obc_space=tangent"   the arm that ran
#   "[obc] real reference set: ... from 117 train records, stride 1"      full reference set
#   "[obc] refresh at epoch 0" ... "rank 13/13"                            (14/14 for obc_affine)
#   "[combos]" and "[cc    ]" lines at each validation                    drift from the G4 values
#   "[params]" at the end                                                  final physical parameters

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Do NOT clear CUDA_VISIBLE_DEVICES here: --gres=gpu:1 sets it to the allocated card, and the
# config refuses a device index for exactly that reason. Clearing it makes device='cuda' fail.

# Still needed with a GPU: data loading, decimation, the window build and the eager validation
# free run are CPU work.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Keep inductor/triton codegen off $HOME (quota) and warm across runs. KEYED BY OS IMAGE: /dataB1
# is shared by heterogeneous nodes, and an unkeyed cache let job 85010 load a GLIBC 2.34 artefact
# on a GLIBC 2.31 node.
CACHE_KEY="$(. /etc/os-release; echo "${ID}${VERSION_ID}")-$(uname -m)"
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache/$CACHE_KEY
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache/$CACHE_KEY
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Optional arm overrides (TR-020/TR-024); empty = the ann_settings.json value.
export NX_ANN="${NX_ANN:-}" NF_SECONDS="${NF_SECONDS:-}" BATCH="${BATCH:-}" CHUNK="${CHUNK:-}"
export EPOCHS="${EPOCHS:-}" LR="${LR:-}" STRIDE="${STRIDE:-}" ITS_PER_VAL="${ITS_PER_VAL:-}"
export FS_TRAIN="${FS_TRAIN:-}" OBC_ARM="${OBC_ARM:-}"
export COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
unset TELICA_SMOKE

echo "cache_key=${CACHE_KEY}"
echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "overrides: OBC_ARM=${OBC_ARM} FS_TRAIN=${FS_TRAIN} NX_ANN=${NX_ANN} NF_SECONDS=${NF_SECONDS} BATCH=${BATCH} CHUNK=${CHUNK} EPOCHS=${EPOCHS} LR=${LR} STRIDE=${STRIDE} ITS_PER_VAL=${ITS_PER_VAL} COMPILE_MODE=${COMPILE_MODE}"
git rev-parse --short HEAD || true
date

echo "=== GPU info (float64 rate: A100 1/2 of float32, RTX cards ~1/32) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "=== host CPU (t_step tracks HOST dispatch, not the GPU) ==="
lscpu | grep -E "CPU\(s\)|Thread|Core|Socket|Model name"

echo "=== toolchain (inductor needs gcc and triton) ==="
which gcc && gcc --version | head -n 1 || echo "gcc NOT FOUND -> inductor will fail"
python -c "import triton; print('triton', triton.__version__)" || echo "triton NOT FOUND"

# The frozen normalisation must exist (it ships with telica-real/); never rebuilt here.
test -f telica-real/pipeline/norm_frozen.json

echo "=== telica-real augmentation ==="
srun --cpu-bind=cores python -u telica-real/pipeline/telica_augment.py

date
echo "done"
