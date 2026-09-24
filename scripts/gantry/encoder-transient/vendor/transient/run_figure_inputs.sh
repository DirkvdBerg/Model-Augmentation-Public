#!/usr/bin/env bash
# The two experiment inputs the presentation figures still need, run STRICTLY ONE AT A TIME.
# The first attempt ran the window sweep and both R2 arms concurrently and all three died, one on
# `CUDA error: unknown error` and two on a host MemoryError. Nothing was wrong with the runs.
set -u
cd "$(dirname "$0")/../../../.."
py () {
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
    -n GraduationProject python -u "$@" 2>&1 | grep --line-buffered -viE "gym|migration|upgrade|replace 'import"
}
echo "=== R2 figure arm: encoder at initialisation"
py scripts/gantry/transient/code/r2_freerun.py --records V1_standstill_Yp10 V3_ysweep_Yp10 \
   --horizon 4000 --n-starts 20 --tag r2fig_init
echo "=== R2 figure arm: encoder trained on the exact state"
py scripts/gantry/transient/code/r2_freerun.py --records V1_standstill_Yp10 V3_ysweep_Yp10 \
   --horizon 4000 --n-starts 20 \
   --encoder scripts/gantry/transient/results/r1_exact_long_encoder.pt --tag r2fig_trained
for NA in 15 59 119; do
  echo "=== window sweep to convergence, na=$NA"
  py scripts/gantry/transient/code/r1_encoder_supervised.py --epochs 1000 --lr 1e-4 \
     --na-nb "$NA" --tag "r1_exact_long_na$NA"
done
echo "=== all figure inputs done"
