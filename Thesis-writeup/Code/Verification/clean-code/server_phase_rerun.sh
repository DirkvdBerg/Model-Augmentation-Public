#!/bin/bash
# D-295: rerun the finished tasks of a server_phase.sh run whose name matches a pattern, then the summary again.
#   bash server_phase_rerun.sh WORK 'NAME_REGEX'          MEM=128gb bash ...: another memory request for these tasks
# Refused while a matching task is still queued or running (squeue). Removes each matching task's --out folder,
# submits those array indices of server_phase.sbatch, cancels the pending summary job and submits a new one that waits
# for every clean-phase and clean-gpu job of this user.

set -euo pipefail
WORK=${1:?usage: bash server_phase_rerun.sh WORK NAME_REGEX}
PAT=${2:?usage: bash server_phase_rerun.sh WORK NAME_REGEX}
cd "$(dirname "$0")"
IDX=() ; OUTS=()
N=0
# array indices of this user's clean-phase tasks still queued or running (any array of this work folder)
BUSY=" $(squeue -u "$USER" -h -r -n clean-phase -o %K | paste -sd' ') "
while IFS=$'\t' read -r NAME ARM COV CMD; do
  if [[ $NAME =~ $PAT ]]; then
    [[ $BUSY == *" $N "* ]] && { echo "$NAME (index $N) is still queued or running; wait or scancel it"; exit 1; }
    IDX+=("$N")
    O=$(sed -n 's/.*--out \([^ ]*\).*/\1/p' <<< "$CMD")
    [ -n "$O" ] && OUTS+=("$O")
  fi
  N=$((N + 1))
done < "$WORK/tasks.tsv"
[ ${#IDX[@]} -gt 0 ] || { echo "no task matches $PAT"; exit 1; }
for O in "${OUTS[@]}"; do
  case $O in "$WORK"/*) rm -rf "$O" ;; *) echo "refusing to remove $O outside $WORK"; exit 1 ;; esac
done
LIST=$(IFS=,; echo "${IDX[*]}")
A=$(sbatch --parsable --array="$LIST" ${MEM:+--mem=$MEM} --export=ALL,WORK="$WORK" server_phase.sbatch)
OLD=$(squeue -u "$USER" -h -n clean-phase-summary -o %i | paste -sd' ')
[ -n "$OLD" ] && scancel $OLD
DEP=$(squeue -u "$USER" -h -n clean-phase,clean-gpu -o %F | sort -u | paste -sd:)
S=$(sbatch --parsable ${DEP:+--dependency=afterany:$DEP} --export=ALL,WORK="$WORK" server_phase_summary.sbatch)
echo "rerun ${#IDX[@]} tasks (indices $LIST) as array $A; summary $S (cancelled: ${OLD:-none})"
