#!/bin/bash
#SBATCH -J polish
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/thesis-results-4/%x_%A_%a.out

# Polish-only rerun of one final-campaign grey-box run (D-276): run_thesis.sbatch with
# Thesis-writeup/Code/Server/greybox/polish_only.py as the program. The array task id is the run's
# runs.tsv row (so the THESIS_* variables are the campaign run's), POLISH_RUN_ID its folder id.
# The partition is chosen at submission: U rows need about 12.4 GB (R4, R7), OBC rows about 28 GB
# (R13), so U on hawaii, lanai or molokai and OBC on lanai or molokai; never oahu (11 GB cards).
#
#   POLISH_RUN_ID=89797 sbatch -p hawaii,lanai,molokai -a 223 Thesis-writeup/Code/Server/greybox/run_polish_only.sbatch
#
# Sanity lines to grep in the output:
#   "[polish-only] config.json identical to the campaign run"
#   "start_phase=lbfgs: skipping Adam"
#   "fixed batch: 16384 of 85572 windows (stride 10 -> 50)"   the campaign's window set
#   "[lbfgs] ACCEPTED" or "[lbfgs] ROLLED BACK", never "ABORTED"

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

: "${POLISH_RUN_ID:?set POLISH_RUN_ID to the run folder id}"
export POLISH_RUN_ID
RUN_TABLE="${RUN_TABLE:-Thesis-writeup/Code/Server/tables/runs.tsv}"
ROW=$(awk -F'\t' -v id="${SLURM_ARRAY_TASK_ID:?submit as an array task whose id is the runs.tsv row}" 'NR>1 && $1==id' "$RUN_TABLE")
if [ -z "$ROW" ]; then echo "no run with id ${SLURM_ARRAY_TASK_ID} in ${RUN_TABLE}"; exit 2; fi
read -r RUN_ID TIER THESIS_ARM THESIS_NOISE THESIS_START THESIS_NA THESIS_SEED THESIS_NF ROW_ARCH ROW_PS2 ROW_INIT <<< "$ROW"
export THESIS_ARM THESIS_NOISE THESIS_START THESIS_NA THESIS_SEED THESIS_NF
if [ -n "$ROW_ARCH" ]; then export THESIS_ARCH="$ROW_ARCH"; fi
if [ -n "$ROW_PS2" ]; then export THESIS_PS2="$ROW_PS2"; fi
if [ -n "$ROW_INIT" ]; then export THESIS_INIT="$ROW_INIT"; fi
echo "polish-only: run_id=${RUN_ID} folder=${POLISH_RUN_ID} tier=${TIER} arm=${THESIS_ARM} noise=${THESIS_NOISE} start=${THESIS_START} na=${THESIS_NA} seed=${THESIS_SEED} nf=${THESIS_NF}"
echo "architecture=${THESIS_ARCH:-mlp} init=${THESIS_INIT:-zero} ps2=${THESIS_PS2:-0}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

CACHE_KEY="$(. /etc/os-release; echo "${ID}${VERSION_ID}")-$(uname -m)"
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache/$CACHE_KEY
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache/$CACHE_KEY
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "job_id=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "node_list=${SLURM_JOB_NODELIST} partition=${SLURM_JOB_PARTITION}"
date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

srun --cpu-bind=cores python -u Thesis-writeup/Code/Server/greybox/polish_only.py

date
echo "done"
