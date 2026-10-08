#!/bin/bash
#SBATCH -J thesis-smokeall
#SBATCH -p hawaii,oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 01:30:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/thesis-results/%x_%j.out

# ONE job that launch-tests every configuration of the campaign: the smoke rows of runs.tsv
# (ids 101-104), run one after another with THESIS_SMOKE=1 (2 s records, eager, 4 updates, a
# 2-iteration polish, then the full end-of-run report). Every row runs even if an earlier one
# fails; the summary at the end gives PASS/FAIL per row. Compilation is not in it (the Karnopp
# runs already compiled on both partitions).
#   101  Aug-OBC  noisy      detuned  2 states  0.1 s   main configuration: OBC, anti-alias, D-229, report
#   102  Aug-U    noise-free correct  0 states  0.1 s   zero added states, unfiltered path, correct start
#   103  Aug-OBC  noisy      detuned  8 states  0.4 s   largest model, longest window, chunked polish
#   104  Aug-OBC  noise-free correct  4 states  0.1 s   the 4-state option
#
# Submit from the repo root (SLURM does not create the log folder):
#   mkdir -p /home/dirk_van_den_berg/logs/augmentation/thesis-results
#   sbatch scripts/gantry/thesis-results/smoke_all.sbatch

set -o pipefail       # deliberately not -e: one failing row must not stop the others

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
CACHE_KEY="$(. /etc/os-release; echo "${ID}${VERSION_ID}")-$(uname -m)"
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache/$CACHE_KEY
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache/$CACHE_KEY
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "job_id=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST} partition=${SLURM_JOB_PARTITION}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "git=$(git rev-parse --short HEAD 2>/dev/null)"
date

RUN_TABLE=scripts/gantry/thesis-results/runs.tsv
export THESIS_SMOKE=1
JOB=${SLURM_JOB_ID}
SUMMARY=()
for ID in $(awk -F'\t' 'NR>1 && $2=="smoke" {print $1}' "$RUN_TABLE"); do
  read -r RUN_ID TIER THESIS_ARM THESIS_NOISE THESIS_START THESIS_NA THESIS_SEED THESIS_NF <<< \
    "$(awk -F'\t' -v id="$ID" 'NR>1 && $1==id' "$RUN_TABLE")"
  export THESIS_ARM THESIS_NOISE THESIS_START THESIS_NA THESIS_SEED THESIS_NF
  echo
  echo "################ smoke ${RUN_ID}: arm=${THESIS_ARM} noise=${THESIS_NOISE} start=${THESIS_START} na=${THESIS_NA} nf=${THESIS_NF} ################"
  T0=$(date +%s)
  # Own run folder per row (the entry names it after SLURM_JOB_ID); plain python, not srun, so
  # the per-row SLURM_JOB_ID does not reach the scheduler.
  SLURM_JOB_ID="${JOB}_smoke${RUN_ID}" python -u scripts/gantry/gantry_interconnect_dynamic.py
  RC=$?
  DT=$(( $(date +%s) - T0 ))
  if [ $RC -eq 0 ]; then S=PASS; else S="FAIL (exit ${RC})"; fi
  SUMMARY+=("smoke ${RUN_ID}  ${THESIS_ARM}/${THESIS_NOISE}/${THESIS_START}/na${THESIS_NA}/nf${THESIS_NF}: ${S}  (${DT} s)")
done

echo
echo "================ SMOKE SUMMARY (job ${JOB}) ================"
printf '%s\n' "${SUMMARY[@]}"
date
