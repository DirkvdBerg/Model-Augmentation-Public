#!/usr/bin/env bash
# R1 window sweep at an EQUAL and CONVERGED budget (D-197, section 19.3).
# The first sweep ran 200 epochs and left dTheta unsettled at na=119, so its line could not be
# put in front of anyone. Same 1000 epochs and same rate for every window; na=29 at 1000 epochs
# already exists as r1_exact_long and is not repeated.
set -u
cd "$(dirname "$0")/../../../.."
for NA in 15 59 119; do
  echo "=================================================================== na=$NA"
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
    -n GraduationProject python -u scripts/gantry/transient/code/r1_encoder_supervised.py \
    --epochs 1000 --lr 1e-4 --na-nb "$NA" --tag "r1_exact_long_na$NA" 2>&1 \
    | grep --line-buffered -viE "gym|migration|upgrade|replace 'import"
done
