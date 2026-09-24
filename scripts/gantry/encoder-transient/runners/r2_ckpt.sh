#!/bin/bash
#SBATCH -J et-r2-ckpt
#SBATCH -p oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=32gb
#SBATCH -t 04:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/encoder-transient/et_r2_ckpt_%j.out
# G5 (b), ET-008. NOT SUBMITTED. CPU is enough (no gradients). CKPT = a trained checkpoint of the
# frictionless production config (the `<base>.pt`+`.npz` pair or a deepSI `_best.pth`), e.g. the
# best checkpoint of `sim_arm.sh ARM=burnin` or of any production run on augmentation_ma50_b140-230_a6_z03:
#     CKPT=/dataB1/.../gantry_ckpt_<job>_best.pth sbatch scripts/gantry/encoder-transient/runners/r2_ckpt.sh
# The checkpoint must match the production block structure (no OBC, no joint estimation).
source scripts/gantry/encoder-transient/runners/_server_env.sh
export CKPT="${CKPT:?set CKPT=<checkpoint>}"
srun --cpu-bind=cores python -u scripts/gantry/encoder-transient/runners/r2_ckpt.py
date; echo done
