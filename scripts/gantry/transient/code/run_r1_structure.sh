#!/usr/bin/env bash
# R1 structure arms (D-197), all at the rate the lr sweep preferred (1e-4):
#   long    is the improvement of the lr sweep a convergence limit or an epoch budget?
#   window  does a longer encoder window buy anything, i.e. is the limit INFORMATION in the
#           window or the map that reads it?
#   fd      what would training against the record's finite-difference velocities give instead?
set -u
cd "$(dirname "$0")/../../../.."
run () {
  echo "=================================================================== $*"
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
    -n GraduationProject python -u scripts/gantry/transient/code/r1_encoder_supervised.py \
    "$@" 2>&1 | grep -viE "gym|migration|upgrade|replace 'import"
}
run --epochs 1000 --lr 1e-4 --tag r1_exact_long
for NA in 15 59 119; do
  run --epochs 200 --lr 1e-4 --na-nb "$NA" --tag "r1_exact_na$NA"
done
run --epochs 200 --lr 1e-4 --target fd --tag r1_fd
