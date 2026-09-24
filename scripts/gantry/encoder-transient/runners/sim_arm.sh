#!/bin/bash
#SBATCH -J et-sim-arm
#SBATCH -p oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/encoder-transient/et_sim_%x_%j.out
# G5 (a), ET-008. NOT SUBMITTED by the session that wrote it. One arm per job, SAME checkout:
#     ARM=burnin    sbatch scripts/gantry/encoder-transient/runners/sim_arm.sh   # reference (K = 100)
#     ARM=g2static  sbatch scripts/gantry/encoder-transient/runners/sim_arm.sh   # G2 map, W^a = 0, K = 0
#     ARM=g2refresh sbatch scripts/gantry/encoder-transient/runners/sim_arm.sh   # model-Jacobian map, K = 0
# Needs outputs/g2/g2_attempt1_encoders.pt (g2static) in the checkout. Predictions: runners/PREDICTIONS.md.
# Grep: "[refresh] update" (one per epoch, g2refresh), "[nf" lines for grow, "Best validation".
# ET-013: the G2 arms freeze the encoder linear map by default; ENC_TRAIN_LINEAR=1 ARM=... is the paired unfrozen arm.
source scripts/gantry/encoder-transient/runners/_server_env.sh
export ARM="${ARM:?set ARM=burnin|g2static|g2refresh}" REFRESH_EVERY="${REFRESH_EVERY:-}"
echo "ARM=${ARM} REFRESH_EVERY=${REFRESH_EVERY:-<one epoch>}"
srun --cpu-bind=cores python -u scripts/gantry/encoder-transient/runners/sim_arm.py
date; echo done
