#!/bin/bash
#SBATCH -J gantry-innov-open
#SBATCH --array=1-2
#SBATCH -p oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/absorber-learning-diagnosis/gantry_innovation_opening_%A_%a.out

# Innovation opening phase (absorber-learning diagnosis, DECISIONS.md E3; local probe E3 = f_inn.log).
# Script: scripts/gantry/absorber-learning-diagnosis/gantry_innovation_opening.py, a copy of
# gantry_interconnect_dynamic.py whose changes are marked "INNOVATION OPENING": for the first INNOV_OPEN
# training loss evaluations the thesis loss gets two temporary terms (one-step equation error on the added
# rows; a training-only head predicting the model's next INNOV_M output errors from the encoder's added
# states), decaying linearly to zero, with a x INNOV_FACTOR learning rate on the ANN's added-state output
# rows, the encoder's W^a and the head; afterwards the run is the thesis run. Model, init, routing unchanged.
# Array: 1 = thesis run 42 settings (arm U), 2 = thesis run 41 settings (arm OBC); both noisy, detuned,
# n_a 2, seed 1, nf 0.1 s; float64, batch 512, lr 1e-5, 2x16 nets, epochs 200, L-BFGS polish, cuda, compile.
#
# Pre-flight (local, must pass first): test_innovation_opening.py ("ALL CHECKS PASSED") and
# smoke_innovation_opening.cmd (both arms, THESIS_SMOKE=1, "=== EXIT 0 ===").
#
# Submit from the repo root:
#   mkdir -p /home/dirk_van_den_berg/logs/augmentation/absorber-learning-diagnosis
#   sbatch scripts/gantry/absorber-learning-diagnosis/run_gantry_innovation_opening.sh
#
# Sanity lines to grep in the output:
#   "[thesis run] arm=u ..." or "arm=obc ..."      the arm
#   "[innovation opening] configured"              the change engaged
#   "[innovation opening] phase ended"             the opening finished (at evaluation INNOV_OPEN)
#   "training rollout COMPILED"                    compilation engaged

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# The run: thesis run 42's variables (runs.tsv id 42).
case "${SLURM_ARRAY_TASK_ID:-1}" in
  1) ARM=u ;;     # thesis run 42
  2) ARM=obc ;;   # thesis run 41
  *) echo "unknown array task ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac
export THESIS_ARM=$ARM THESIS_NOISE=noisy THESIS_START=detuned THESIS_NA=2 THESIS_SEED=1 THESIS_NF=0.1
export INNOV_OPEN=2000 INNOV_M=40 INNOV_FACTOR=100
echo "array_task=${SLURM_ARRAY_TASK_ID:-1} arm=${ARM} innov_open=${INNOV_OPEN} innov_m=${INNOV_M} innov_factor=${INNOV_FACTOR}"

# Do NOT clear CUDA_VISIBLE_DEVICES here: --gres=gpu:1 sets it to the allocated card, and the
# config refuses a device index for exactly that reason. Clearing it makes device='cuda' fail.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Keep inductor/triton codegen off $HOME (quota) and warm across runs, KEYED BY OS IMAGE (job 85010
# died on blade2 importing a launcher an A100 job had compiled against a newer GLIBC).
CACHE_KEY="$(. /etc/os-release; echo "${ID}${VERSION_ID}")-$(uname -m)"
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache/$CACHE_KEY
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache/$CACHE_KEY
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
echo "cache_key=${CACHE_KEY}"

echo "job_id=${SLURM_JOB_ID} array_job_id=${SLURM_ARRAY_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
date

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "=== toolchain (inductor needs gcc and triton) ==="
which gcc && gcc --version | head -n 1 || echo "gcc NOT FOUND -> inductor will fail"
python -c "import triton; print('triton', triton.__version__)" || echo "triton NOT FOUND"

echo "=== git (the code this run used) ==="
git rev-parse --short HEAD || true

echo "=== gantry_innovation_opening (thesis settings, arm ${ARM}, innovation opening) ==="
srun --cpu-bind=cores python -u scripts/gantry/absorber-learning-diagnosis/gantry_innovation_opening.py

date
echo "done"
