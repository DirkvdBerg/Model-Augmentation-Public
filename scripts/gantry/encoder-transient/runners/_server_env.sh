# Shared server environment for the encoder-transient runners (ET-008). Sourced, not submitted.
# Copied from telica-real/pipeline/runners/server_run.sh (the user's working GPU conventions).
set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
CACHE_KEY="$(. /etc/os-release; echo "${ID}${VERSION_ID}")-$(uname -m)"
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache/$CACHE_KEY
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache/$CACHE_KEY
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
unset DRY SMOKE R2_SMOKE TELICA_SMOKE
echo "job_id=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST} cuda=${CUDA_VISIBLE_DEVICES:-<unset>}"
git rev-parse --short HEAD || true
date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
# The frictionless dataset and the exact-truth cache must be on the server:
test -d data/gantry/matlab/trajectory/augmentation_ma50_b140-230_a6_z03
# r2_ckpt and DRY runs read the exact truth: sync scripts/gantry/transient/cache/ (62 MB) or it is rebuilt.
ls scripts/gantry/transient/cache/*.npz >/dev/null || echo "WARNING: exact-truth cache missing, it will be rebuilt (slow)"
