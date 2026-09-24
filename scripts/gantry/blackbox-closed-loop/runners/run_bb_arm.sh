#!/bin/bash
#SBATCH -J bbcl-arm
#SBATCH -p molokai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/blackbox-closed-loop/bbcl_arm%j.out

# ONE ARM of the closed-loop black box (DECISIONS.md BB-010). NOT SUBMITTED by the session that
# wrote it. Header, environment and cache handling copied from the grey box's own runner
# (scripts/gantry/orthogonal-by-construction/runners/run_obc_arm.sh), so the two jobs run on the
# same partition, card type, CPU count, memory and wall clock.
#
#     BB_ARM=<arm from REPORT.md G3>  sbatch scripts/gantry/blackbox-closed-loop/runners/run_bb_arm.sh
#     BB_ARM=random                   sbatch scripts/gantry/blackbox-closed-loop/runners/run_bb_arm.sh
#
# `random` is the Jan-faithful control and always runs. Every hyperparameter except the black box
# itself comes from CFG in scripts/gantry/gantry_interconnect_dynamic.py at run time; the only two
# CFG fields replaced are lr and adam_eps (BB-004/BB-005). Submit every arm from the SAME unedited
# checkout, and from the checkout the grey box's comparison run uses: editing CFG between the
# launches is how a paired comparison silently stops being paired.
#
# Wall clock: see REPORT.md G4 for the projected cost of CFG's update budget. If it exceeds the
# 24 h here, raise -t (check the partition limit) rather than lowering epochs, or the update budget
# is no longer the grey box's.
#
# Sanity lines to grep in the log:
#   "BB_ARM="                                  which arm
#   "<- CFG"                                    exactly two lines: lr and adam_eps
#   "[closed loop] one bank over 24 records"   the grey box's controller bank
#   "training rollout COMPILED"                compilation engaged (CFG compile_mode)
#   "[bb] arm="                                black-box structure line
#   "[bb] val closed-loop sim-RMS"             final selection-set score, same harness
#   "[bb] test closed-loop sim-RMS"            test-set score for the comparison table

set -eo pipefail

if [ -z "${BB_ARM:-}" ]; then
    echo "BB_ARM is not set. This runner submits ONE arm and will not guess which."
    echo "  BB_ARM=random|n4sid|frf sbatch $0"
    exit 2
fi
case "$BB_ARM" in
    random|n4sid|frf) ;;
    *) echo "BB_ARM must be random, n4sid or frf, got '${BB_ARM}'"; exit 2 ;;
esac
export BB_ARM

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Same OS-keyed inductor/triton cache as the grey box's runner (job 85010 GLIBC incident).
CACHE_KEY="$(. /etc/os-release; echo "${ID}${VERSION_ID}")-$(uname -m)"
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache/$CACHE_KEY
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache/$CACHE_KEY
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
echo "cache_key=${CACHE_KEY}"

echo "BB_ARM=${BB_ARM}"
echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
date
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true
which gcc && gcc --version | head -n 1 || echo "gcc NOT FOUND -> inductor will fail"

echo "=== EFFECTIVE configuration snapshot (BB_ARM=${BB_ARM}) ==="
python -u scripts/gantry/blackbox-closed-loop/runners/train_bb.py --print-config || true

echo "=== black box, closed-loop training, arm ${BB_ARM} ==="
srun --cpu-bind=cores python -u scripts/gantry/blackbox-closed-loop/runners/train_bb.py

date
echo "done ${BB_ARM}"
