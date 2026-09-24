#!/bin/bash
#SBATCH -J et-telica-arm
#SBATCH -p oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/encoder-transient/et_telica_%j.out
# G5 (c), ET-008 / ET-012. NOT SUBMITTED. The telica-real production arm (5 kHz, nf 200, batch 256, nx_ann 8,
# fixed G4 baseline + ANN) with the G4 attempt-3 encoder (RFS, na 7) swapped in; otherwise identical to
# telica-real/pipeline/runners/server_run.sh, so the paired reference is that script's default run.
#     sbatch scripts/gantry/encoder-transient/runners/telica_arm.sh
source scripts/gantry/encoder-transient/runners/_server_env.sh
test -f scripts/gantry/encoder-transient/telica/vendor_telica/pipeline/norm_frozen.json
export ET_G2_ENCODER=scripts/gantry/encoder-transient/outputs/telica/g4/g2_telica_a3_encoder.pt
export ET_NA_NB=7                     # ET-012: the G4 attempt-3 window (1.6 ms at 5 kHz)
test -f "$ET_G2_ENCODER"
export COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
srun --cpu-bind=cores python -u scripts/gantry/encoder-transient/telica/vendor_telica/pipeline/telica_augment.py
date; echo done
