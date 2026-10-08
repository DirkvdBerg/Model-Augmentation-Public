#!/bin/bash
#SBATCH -J thesis
#SBATCH -p hawaii,oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/live-init-g-aug/%x_%A_%a.out

# One run of the thesis campaign (Thesis-writeup/Documentation/TRAINING-DESIGN.md section 4).
# Submit through submit.sh, never directly: it passes RUN_TABLE and runs as an array whose task
# id is the run id in runs.tsv. This file supplies the allocation, the environment and the run's
# six variables; every hyperparameter comes from the THESIS block in gantry_interconnect_dynamic.py
# (D-222), whose "Configuration:" print in the log is authoritative.
#
# Expected config: float64, nf=400 (0.1 s; sweep 800 / 1600), batch 512, lr 1e-5, 16x2 tanh,
#   na_nb=29, nx_ann = THESIS_NA, epochs 200, stride 10, L-BFGS polish 100 iterations history 50,
#   compile_mode=reduce-overhead, device=cuda.
#
# Sanity lines to grep in the output:
#   "[thesis run] arm=..."                 the run the task id selected
#   "one bank over 24 records, 1 distinct controllers"   K1 for every record (D-221)
#   "training rollout COMPILED"            compilation engaged
#   NO "skipping cudagraphs due to cpu"    CUDA graphs kept their fast path

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# The run: the row of RUN_TABLE whose id is this array task.
RUN_TABLE="${RUN_TABLE:-scripts/gantry/thesis-results/runs.tsv}"
ROW=$(awk -F'\t' -v id="${SLURM_ARRAY_TASK_ID:?submit through submit.sh}" 'NR>1 && $1==id' "$RUN_TABLE")
if [ -z "$ROW" ]; then echo "no run with id ${SLURM_ARRAY_TASK_ID} in ${RUN_TABLE}"; exit 2; fi
read -r RUN_ID TIER THESIS_ARM THESIS_NOISE THESIS_START THESIS_NA THESIS_SEED THESIS_NF ROW_ARCH ROW_PS2 <<< "$ROW"
export THESIS_ARM THESIS_NOISE THESIS_START THESIS_NA THESIS_SEED THESIS_NF
# Comparison rows specify both choices; legacy eight-column tables retain env defaults.
if [ -n "$ROW_ARCH" ]; then export THESIS_ARCH="$ROW_ARCH"; fi
if [ -n "$ROW_PS2" ]; then export THESIS_PS2="$ROW_PS2"; fi
echo "run_id=${RUN_ID} tier=${TIER} arm=${THESIS_ARM} noise=${THESIS_NOISE} start=${THESIS_START} na=${THESIS_NA} seed=${THESIS_SEED} nf=${THESIS_NF} smoke=${THESIS_SMOKE:-0} nits=${THESIS_NITS:-<epochs>}"
echo "architecture=${THESIS_ARCH:-resnet} ps2=${THESIS_PS2:-0}"

# Do NOT clear CUDA_VISIBLE_DEVICES here: --gres=gpu:1 sets it to the allocated card, and the
# config refuses a device index for exactly that reason. Clearing it makes device='cuda' fail.

# Still needed with a GPU: data loading, the window build and the eager validation free run are
# all CPU work, and validation is the single largest per-epoch cost once training is compiled.
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

echo "job_id=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "node_list=${SLURM_JOB_NODELIST} partition=${SLURM_JOB_PARTITION}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
date

echo "=== GPU info (float64 speed depends on the card) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "=== host CPU (t_step tracks HOST dispatch, not the GPU) ==="
lscpu | grep -E "CPU\(s\)|Thread|Core|Socket|Model name"

echo "=== toolchain (inductor needs gcc and triton) ==="
which gcc && gcc --version | head -n 1 || echo "gcc NOT FOUND -> inductor will fail"
python -c "import triton; print('triton', triton.__version__)" || echo "triton NOT FOUND"

echo "=== git (the code this run used) ==="
git rev-parse --short HEAD || true
git status --short -- scripts/gantry/gantry_interconnect_dynamic.py scripts/gantry/gantry_dynamic || true

echo "=== gantry_interconnect_dynamic, thesis run ${RUN_ID} ==="
srun --cpu-bind=cores python -u scripts/gantry/gantry_interconnect_dynamic.py

date
echo "done"
