#!/bin/bash
#SBATCH -J clean-timing
#SBATCH -p hawaii,oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 12:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/clean-code/%x_%j.out

# Compiled timing of D-284 step 5 (timing_driver.py): arms mainline, pre-round and final snapshot, one GPU
# allocation, every pass its own process. Three timing passes per arm, interleaved (mainline, pre-round, final,
# then again) so drift over the job falls on every arm alike. One compile cache per arm: cold for its first pass
# (the compile and CUDA-graph seconds come from that pass), warm for its two other passes and its traced
# diagnostic pass (TORCH_TRACE, TORCH_LOGS=graph_breaks,recompiles; 20 warm-up and 10 traced updates), which run last.
# Every pass writes its own result.json, so a job cut by the time limit keeps what finished.
# Afterwards: python -u timing_driver.py --summarise $OUT, and tlparse on each $OUT/trace-<arm>.

set -o pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
REPO=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
cd "$REPO/Thesis-writeup/Code/Verification/clean-code"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HARNESS_THREADS=$SLURM_CPUS_PER_TASK
OUT="${OUT:-$REPO/simulations/clean-golden/timing-D284}"
CACHE="/dataB1/dirk_van_den_berg/timing-cache-D284/$SLURM_JOB_ID"
ARMS=(mainline pre-round final)
mkdir -p "$OUT/logs"
echo "out=${OUT} node=${SLURM_JOB_NODELIST} cuda=${CUDA_VISIBLE_DEVICES:-<unset>}"; date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

one_pass() {   # $1 arm, $2 pass name, then extra driver arguments
  local arm=$1 name=$2; shift 2
  export TORCHINDUCTOR_CACHE_DIR="$CACHE/$arm/inductor" TRITON_CACHE_DIR="$CACHE/$arm/triton"
  if [ -d "$CACHE/$arm" ]; then export TIMING_CACHE=warm; else export TIMING_CACHE=cold; fi
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  echo "=== pass $name ($arm, cache $TIMING_CACHE) $(date)"
  python -u timing_driver.py --arm "$arm" --out "$OUT/$name" "$@" 2>&1 | tee "$OUT/logs/$name.log"
  echo "=== pass $name exit $? $(date)"
}

for REP in 1 2 3; do
  for ARM in "${ARMS[@]}"; do
    one_pass "$ARM" "$ARM-r$REP"
  done
done
for ARM in "${ARMS[@]}"; do
  TORCH_TRACE="$OUT/trace-$ARM" TORCH_LOGS=graph_breaks,recompiles one_pass "$ARM" "$ARM-trace" --timed 10
done
rm -rf "$CACHE"
python -u timing_driver.py --summarise "$OUT" 2>&1 | tee "$OUT/logs/summary.log"
date; echo "done timing"
