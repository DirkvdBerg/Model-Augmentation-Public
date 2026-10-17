#!/bin/bash
#SBATCH -J thesis-smokeall
#SBATCH -p hawaii,oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 02:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/thesis-results-3/%x_%j.out

# ONE job that launch-tests every row type of the thesis campaign (THESIS-RESULTS.md section 4):
# the tier "thesis-smoke" rows of runs.tsv, run one after another with THESIS_SMOKE=1 (2 s records,
# eager, 4 updates, a 2-iteration polish, then the full end-of-run report). Every row runs even if
# an earlier one fails; the summary at the end gives PASS/FAIL per row. Compilation is not in it:
# watch the first real run's log for "training rollout COMPILED".
#   211  Aug-OBC  lowpass  detuned  2 states  PS2    mirrors 201 (R1)
#   212  Aug-OBC  lowpass  correct  2 states  PS2    mirrors 202 (R2)
#   213  Aug-U    lowpass  correct  2 states  plain  mirrors 203 (R3)
#   214  Aug-U    lowpass  detuned  2 states  PS2    mirrors 204 (R4; 205 differs only in the start)
#   216  Aug-U    lowpass  correct  0 states  plain  mirrors 206 (R6)
#   217  Aug-U    lowpass  correct  8 states  PS2    mirrors 207 (R7)
# All rows: mlp + zero g_aug, seed 1, window 0.1 s. Per row the log must read g_aug=mlp/zero and
# PS2|plain as listed; PS2 rows print "[ps2] opening completed" or a fall-back reason.
#
# Submit from the repo root (SLURM does not create the log folder):
#   mkdir -p /home/dirk_van_den_berg/logs/augmentation/thesis-results-3
#   sbatch Thesis-writeup/Code/Server/greybox/smoke_all.sbatch

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

RUN_TABLE=Thesis-writeup/Code/Server/tables/runs.tsv
export THESIS_SMOKE=1
JOB=${SLURM_JOB_ID}
SUMMARY=()
# SMOKE_TIER selects the rows: thesis-smoke (2026-10-11 rows 211-217) or final-smoke (231-240).
SMOKE_TIER=${SMOKE_TIER:-thesis-smoke}
for ID in $(awk -F'\t' -v t="$SMOKE_TIER" 'NR>1 && $2==t {print $1}' "$RUN_TABLE"); do
  read -r RUN_ID TIER THESIS_ARM THESIS_NOISE THESIS_START THESIS_NA THESIS_SEED THESIS_NF \
    THESIS_ARCH THESIS_PS2 THESIS_INIT <<< "$(awk -F'\t' -v id="$ID" 'NR>1 && $1==id' "$RUN_TABLE")"
  export THESIS_ARM THESIS_NOISE THESIS_START THESIS_NA THESIS_SEED THESIS_NF \
    THESIS_ARCH THESIS_PS2 THESIS_INIT
  echo
  echo "################ smoke ${RUN_ID}: arm=${THESIS_ARM} noise=${THESIS_NOISE} start=${THESIS_START} na=${THESIS_NA} nf=${THESIS_NF} g_aug=${THESIS_ARCH}/${THESIS_INIT} ps2=${THESIS_PS2} ################"
  T0=$(date +%s)
  # Own run folder per row (the entry names it after SLURM_JOB_ID); plain python, not srun, so
  # the per-row SLURM_JOB_ID does not reach the scheduler.
  SLURM_JOB_ID="${JOB}_smoke${RUN_ID}" python -u scripts/gantry/gantry_interconnect_dynamic.py
  RC=$?
  DT=$(( $(date +%s) - T0 ))
  if [ $RC -eq 0 ]; then S=PASS; else S="FAIL (exit ${RC})"; fi
  SUMMARY+=("smoke ${RUN_ID}  ${THESIS_ARM}/${THESIS_NOISE}/${THESIS_START}/na${THESIS_NA}/nf${THESIS_NF}/${THESIS_ARCH}-${THESIS_INIT}/ps2=${THESIS_PS2}: ${S}  (${DT} s)")
done

echo
echo "================ SMOKE SUMMARY (job ${JOB}) ================"
printf '%s\n' "${SUMMARY[@]}"
date
