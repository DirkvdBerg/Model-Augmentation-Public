#!/usr/bin/env bash
# R1 learning-rate arms (D-197). One knob differs between them, the rate, because Adam's step is
# scale-free and the reconstructability init is accurate to a relative 2e-05 on X and Y: the rate
# decides whether training can improve the init at all before it destroys it.
set -u
cd "$(dirname "$0")/../../../.."
for LR in 1e-4 1e-5 1e-6; do
  echo "=================================================================== lr=$LR"
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
    -n GraduationProject python -u scripts/gantry/transient/code/r1_encoder_supervised.py \
    --epochs 200 --lr "$LR" --tag "r1_exact_lr$LR" 2>&1 | grep -viE "gym|migration|upgrade|replace 'import"
done
