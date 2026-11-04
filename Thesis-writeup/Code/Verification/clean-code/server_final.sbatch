#!/bin/bash
#SBATCH -J clean-final
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/clean-code/%x_%A_%a.out

# D-295 final batch: one line of $WORK/$TASKS (written by server_final.sh) per array index: name, tree, command.
# tree final = $COPY-final (Clean/ as it is), pinv = $COPY-pinv (tree B: only the P^-T of D-294 reverted), none = no
# clean code. Partition, cores, memory, GPU and time come from server_final.sh's sbatch call.

set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
REPO=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
cd "$REPO/Thesis-writeup/Code/Verification/clean-code"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
: "${WORK:?submit through server_final.sh}" "${TASKS:?submit through server_final.sh}"
source "$WORK/settings.sh"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$WORK/$TASKS")
IFS=$'\t' read -r NAME TREE CMD <<< "$LINE"
LOG="$WORK/logs/$NAME.log"
{
  echo "task=$NAME tree=$TREE node=$SLURM_JOB_NODELIST"; date
  if [ "$TREE" != none ]; then
    export HARNESS_CLEAN_ROOT="$COPY-$TREE"
    (cd "$HARNESS_CLEAN_ROOT" && sha256sum -c --quiet "$WORK/clean-$TREE.sha256") && echo "clean-$TREE matches its manifest"
  fi
  set +e
  bash -c "$CMD"
  RC=$?
  date; echo "TASK_DONE $NAME rc=$RC"
} > "$LOG" 2>&1
exit $RC
