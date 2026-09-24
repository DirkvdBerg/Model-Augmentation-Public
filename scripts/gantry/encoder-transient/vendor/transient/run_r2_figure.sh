#!/usr/bin/env bash
# R2 re-run for the figure only: per-channel error traces (the first pass averaged the channel
# axis away, which hid Y, the axis the comparison is about). Two records, two encoders.
set -u
cd "$(dirname "$0")/../../../.."
for ARM in init trained; do
  EXTRA=""
  if [ "$ARM" = "trained" ]; then
    EXTRA="--encoder scripts/gantry/transient/results/r1_exact_long_encoder.pt"
  fi
  echo "=================================================================== $ARM"
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
    -n GraduationProject python -u scripts/gantry/transient/code/r2_freerun.py \
    --records V1_standstill_Yp10 V3_ysweep_Yp10 --horizon 4000 --n-starts 20 \
    $EXTRA --tag "r2fig_$ARM" 2>&1 | grep --line-buffered -viE "gym|migration|upgrade|replace 'import"
done
