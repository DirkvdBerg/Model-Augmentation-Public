#!/bin/bash
#SBATCH -J jh-arm
#SBATCH -p molokai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/joint-estimation-horizon/jh_arm%j.out

# ONE ARM of the joint-estimation-horizon G3 pair (DECISIONS.md JH-006, JH-007). NOT SUBMITTED by
# the session that wrote it. Adapted from scripts/gantry/orthogonal-by-construction/runners/
# run_obc_arm.sh; every hyperparameter comes from the VENDORED entry file
# scripts/gantry/joint-estimation-horizon/vendor/gantry_interconnect_dynamic.py, whose JH_ARM
# block pins run 84033's configuration (14 records of augmentation_ma50_b140-230_a6_z03, tangent
# OBC, prior off, 150 epochs) and differs between the arms in exactly one thing:
#
#     JH_ARM=control sbatch scripts/gantry/joint-estimation-horizon/runners/run_jh_arm.sh
#         84033 as it was: detuned start, the ten combinations trained jointly with the ANN
#     JH_ARM=staged  sbatch scripts/gantry/joint-estimation-horizon/runners/run_jh_arm.sh
#         the ten combinations FIXED at the stage-1 fit (JH_STAGE1_JSON), still protected by OBC
#
# Submit both from the SAME unedited checkout. The stage-1 file is written by the session
# (staged/g2_staged.py) and must be synced with the folder.
#
# Expected config (the snapshot below is authoritative): device=cuda compile_mode=reduce-overhead
#   nf=400 batch=512 lr=1e-5 nx_ann=8 24x3 na_nb=29 epochs=150 its_per_val=1300 stride=10
#   float32 joint_estimation=True reduced param_prior=False obc=True tangent lbfgs=True
# Cost (84033 measured): ~10.4 h per arm at epochs=150 on the A100, incl. ~1 h CUDA-graph recording
# before the second update (NOT a hang) and 619 s per validation.
#
# Sanity lines to grep:
#   "JH_ARM="                              which arm
#   "[JH_ARM=staged] ten combinations FIXED" staged arm only; the printed values = stage-1 json
#   "[obc] reference set:"                 671944 tuples from 14 records
#   "[obc] refresh at epoch 0"             rank 10 of 10
#   "training rollout COMPILED"

set -eo pipefail

if [ -z "${JH_ARM:-}" ]; then
    echo "JH_ARM is not set: control or staged."; exit 2
fi
case "$JH_ARM" in
    control|staged) ;;
    *) echo "JH_ARM must be control or staged, got '${JH_ARM}'"; exit 2 ;;
esac
export JH_ARM
export JH_FILESET=obc14
JH=scripts/gantry/joint-estimation-horizon
export JH_STAGE1_JSON=${JH_STAGE1_JSON:-$JH/runners/stage1.json}
if [ "$JH_ARM" = "staged" ] && [ ! -f "$JH_STAGE1_JSON" ]; then
    echo "staged arm needs the stage-1 file $JH_STAGE1_JSON"; exit 2
fi

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
CACHE_KEY="$(. /etc/os-release; echo "${ID}${VERSION_ID}")-$(uname -m)"
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache/$CACHE_KEY
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache/$CACHE_KEY
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "JH_ARM=${JH_ARM}  JH_FILESET=${JH_FILESET}  JH_STAGE1_JSON=${JH_STAGE1_JSON}"
echo "job_id=${SLURM_JOB_ID} node_list=${SLURM_JOB_NODELIST}"
date
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true
[ -f "$JH_STAGE1_JSON" ] && cat "$JH_STAGE1_JSON"

echo "=== EFFECTIVE imported configuration snapshot (JH_ARM=${JH_ARM}) ==="
python -u -c "
import sys; sys.path.insert(0, '$JH/vendor')
from dataclasses import fields
import gantry_interconnect_dynamic as G
for f in sorted(fields(G.CFG), key=lambda f: f.name):
    print('  %-26s %r' % (f.name, getattr(G.CFG, f.name)))
print('  %-26s %r' % ('DERIVED nf', G.CFG.nf))
" || true

srun --cpu-bind=cores python -u $JH/vendor/gantry_interconnect_dynamic.py
date
echo "done ${JH_ARM}"
