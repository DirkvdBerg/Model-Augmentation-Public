#!/bin/bash
#SBATCH -J clean-phase
#SBATCH -p kauai,molokai,lanai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=32gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/clean-code/%x_%A_%a.out

# D-295: one task of $WORK/tasks.tsv (written by server_phase.sh) per array index, one thread (bitwise).
# Arm A imports $WORK/clean-reverted (the D-294 lines reverted), arm B $WORK/clean-final; each is refused unless the
# copy still matches its manifest and is in its D-294 state.

set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
REPO=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
cd "$REPO/Thesis-writeup/Code/Verification/clean-code"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${WORK:?submit through server_phase.sh}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$WORK/tasks.tsv")
IFS=$'\t' read -r NAME ARM COV CMD <<< "$LINE"
LOG="$WORK/logs/$NAME.log"
{
  echo "task=$NAME arm=$ARM coverage=$COV node=$SLURM_JOB_NODELIST"; date
  TREE=$([ "$ARM" = A ] && echo reverted || echo final)
  export HARNESS_CLEAN_ROOT="$WORK/clean-$TREE"
  (cd "$HARNESS_CLEAN_ROOT" && sha256sum -c --quiet "$WORK/clean-$TREE.sha256") && echo "clean-$TREE matches its manifest"
  python -u d294_revert.py status --expect "$TREE"
  set +e                     # a failing task still writes its TASK_DONE line with the exit code
  if [ "$COV" = 1 ]; then
    # shellcheck disable=SC2086
    python -m coverage run --data-file="$WORK/cov/.coverage.$NAME" --include="$HARNESS_CLEAN_ROOT/*" $CMD
  else
    # shellcheck disable=SC2086
    python -u $CMD
  fi
  RC=$?
  date; echo "TASK_DONE $NAME rc=$RC"
} > "$LOG" 2>&1
exit $RC
