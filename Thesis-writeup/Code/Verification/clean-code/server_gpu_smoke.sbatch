#!/bin/bash
#SBATCH -J clean-gpu
#SBATCH -p hawaii,oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 12:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/clean-code/%x_%A_%a.out

# Compiled GPU smokes of the clean-code harness (tasks/clean-code/design.md 6.4 (a) and (b)).
# Array task 0 = R1 (221), task 1 = R6 (226), both lowpass. Both codes run in the SAME job, so on the
# same node and card: (a) mainline then clean; task 1 then runs (b), the compiled polish-only start
# from each side's own (a) `_pre_lbfgs` pair. The one-off CUDA-graph recording took 55 to 61 min in
# production, so expect about an hour per (a) run. Afterwards: python -u compare_gpu.py $OUT

set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
REPO=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
cd "$REPO/Thesis-writeup/Code/Verification/clean-code"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HARNESS_THREADS=$SLURM_CPUS_PER_TASK
CACHE_KEY="$(. /etc/os-release; echo "${ID}${VERSION_ID}")-$(uname -m)"
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache/$CACHE_KEY
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache/$CACHE_KEY
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
OUT="${OUT:-$REPO/simulations/clean-golden/gpu-smoke}"
ROWS=(221 226)
ROW=${ROWS[${SLURM_ARRAY_TASK_ID:?submit with --array=0-1}]}
echo "row=${ROW} out=${OUT} node=${SLURM_JOB_NODELIST} cuda=${CUDA_VISIBLE_DEVICES:-<unset>}"; date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
mkdir -p "$OUT/logs"
for SIDE in mainline clean; do
  python -u gpu_smoke.py --side "$SIDE" --row "$ROW" --out "$OUT/a-$ROW-$SIDE" 2>&1 | tee "$OUT/logs/a-$ROW-$SIDE.log"
done
if [ "$ROW" = 226 ]; then
  for SIDE in mainline clean; do
    if [ "$SIDE" = mainline ]; then SRC="$OUT/a-226-mainline/gpuA-226-lowpass"
    else SRC="$OUT/a-226-clean/thesis_taf_lowpass_noise_data_lsq/gpuA-226-lowpass"; fi
    python -u gpu_smoke.py --side "$SIDE" --row 226 --polish-from "$SRC" --out "$OUT/b-226-$SIDE" 2>&1 \
      | tee "$OUT/logs/b-226-$SIDE.log"
  done
fi
date; echo "done ${ROW}"
