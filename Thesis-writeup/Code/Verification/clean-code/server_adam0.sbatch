#!/bin/bash
#SBATCH -J clean-adam0
#SBATCH -p kauai,molokai,lanai,oahu,hawaii
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=32gb
#SBATCH -t 24:00:00
#SBATCH --array=0-2
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/clean-code/%x_%A_%a.out

# D-295: the Adam resume of the final tree from ZERO moments, as mainline resumes from a thesis deepSI dump (it does
# not read the dump's Adam state), so the two Adam trajectories start alike; then bitwise against mainline's adam
# output of the same work folder. Submit with WORK set: sbatch --export=ALL,WORK=<work folder> server_adam0.sbatch

set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
REPO=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
cd "$REPO/Thesis-writeup/Code/Verification/clean-code"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2
: "${WORK:?submit with --export=ALL,WORK=<work folder>}"
source "$WORK/settings.sh"
RUNS=(R1 R4 R6)
X=${RUNS[$SLURM_ARRAY_TASK_ID]}
export HARNESS_CLEAN_ROOT="$COPY-final" EXPECT_TREE=final
(cd "$HARNESS_CLEAN_ROOT" && sha256sum -c --quiet "$WORK/clean-final.sha256") && echo "clean-final matches its manifest"
LOG="$WORK/logs/adam0-B-clean-$X.log"
python -u phase_run.py --side clean --phase adam --run "$X" --staging "$WORK/staging" --dumps "$HOME/.deepSI/checkpoints" \
  --out "$WORK/adam0/B-clean-$X" --n-its 50 --zero-moments > "$LOG" 2>&1
echo "TASK_DONE adam0-B-clean-$X rc=0" >> "$LOG"
# the mainline side resumed from zero moments already (its log says so); wait for it if it is still running
until grep -q "^TASK_DONE adam-A-mainline-$X rc=" "$WORK/logs/adam-A-mainline-$X.log" 2>/dev/null; do sleep 60; done
python -u compare_phase.py bitwise "$WORK/adam/A-mainline-$X" "$WORK/adam0/B-clean-$X" \
  "$WORK/reports/adam0-B-bitwise-$X.json" | tee -a "$LOG" | grep COMPARE_PHASE
