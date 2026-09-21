#!/bin/bash
#SBATCH -J obc-arm
#SBATCH -p molokai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/orthogonal-by-construction/obc_arm%j.out

# ONE ARM of the D-192/D-200 projection comparison. Same entry point as
# gantry_interconnect_dynamic_gpu.sh;
# every hyperparameter comes from CFG in gantry_interconnect_dynamic.py, NOT from here. The only
# thing this file chooses is the arm, through OBC_ARM, which the entry file validates against an
# allow-list.
#
#     OBC_ARM=noproj sbatch scripts/gantry/orthogonal-by-construction/runners/run_obc_arm.sh
#     OBC_ARM=obc    sbatch scripts/gantry/orthogonal-by-construction/runners/run_obc_arm.sh
#     OBC_ARM=obc_affine sbatch scripts/gantry/orthogonal-by-construction/runners/run_obc_arm.sh
#
# All arms set joint_estimation=True and param_prior=False. Control -> tangent changes only `obc`;
# tangent -> affine changes only `obc_space`.
# Neither is an existing baseline: no run on this dataset has used joint_estimation=True, so the
# control is new too. Submit both from the SAME unedited checkout; editing CFG between the two
# launches is how a paired experiment silently stops being paired.
#
# RUN run_obc_probe.sh FIRST. It settles two things this run cannot recover from: the parameter
# prior (D-193) and whether one production-shape update fits on the card. If the probe reports
# thin headroom, raise checkpoint_chunk in CFG (exact, not truncated, about +33% compute).
# Before the affine arm, also run `probe_compile.py --mode reduce-overhead --space affine` on the
# target GPU: the existing ladder qualifies the tangent arm, while affine threads a third item.
#
# Expected config (verify against the run's own snapshot below, which is authoritative):
#   device=cuda  compile_mode=reduce-overhead  nf=400  batch=512  lr=1e-5  nx_ann=8  24x3
#   na_nb=29  epochs=150  its_per_val=1300  stride=10  chunk=0  float32  joint_estimation=True
#   physics_parameterization=reduced  param_prior=False  combo_init_detune=10 percent
#
# WHAT OBC COSTS, measured on a Quadro P2000 at batch 512, eager (implementation/results/
# 2026-09-16/): 2.7x to 3.3x per training step, 3.8x to 4.0x per update, plus a once-per-epoch
# basis refresh of about 11 s and a once-per-objective reference pass and solve of about 0.09 s.
#
# WHAT IT COSTS COMPILED, measured on the A100 (jobs 83962 and 83978, both OBC, nf=400):
#   one-off   55 to 61 min of CUDA-graph recording before the SECOND update completes. This is
#             NOT a hang. The first update looks normal (~100 s), then the progress bar sits for
#             an hour. Two earlier jobs were cancelled during it and wrongly written off.
#   steady    1.24 s per update, against 1.89 s on compile_mode='default'.
#   per val   619 s, which is why its_per_val is 1300 and not 650.
# Budget at epochs=150: 6.8 h stepping + 1 h recording + 2.6 h validation = about 10.4 h.
#
# Sanity lines to grep:
#   "OBC_ARM="                             which arm this job is
#   "[obc] reference set:"                 671944 tuples from 14 records
#   "[obc] refresh at epoch 0"             rank 10 of 10, and the condition number
#   "training rollout COMPILED"            compilation engaged
#   NO "skipping cudagraphs due to cpu"    CUDA graphs kept their fast path
#   "[nf val ] rms ..."                    the window probe ran (not "failed (non-fatal)")
#
# NO --signal=USR1@1800: nothing installs a SIGUSR1 handler and Python's default action for it
# is to terminate, so the job would be killed at 23:30 and lose the results NPZ, the baselines
# and the diagnostics. Checkpoints survive, since they are written at every improving validation.
# (This used to read "the only line dropped from the gantry_interconnect_dynamic_gpu.sh
# template". That template carried the bug until 2026-09-21, when the line was removed there
# too, so the two runners no longer differ on this point.)

set -eo pipefail

if [ -z "${OBC_ARM:-}" ]; then
    echo "OBC_ARM is not set. This runner submits ONE arm and will not guess which."
    echo "  OBC_ARM=noproj sbatch $0    joint estimation, no projection"
    echo "  OBC_ARM=obc    sbatch $0    joint estimation, ten-column tangent OBC"
    echo "  OBC_ARM=obc_affine sbatch $0  joint estimation, eleven-column affine OBC"
    exit 2
fi
case "$OBC_ARM" in
    noproj|obc|obc_affine) ;;
    *) echo "OBC_ARM must be 'noproj', 'obc', or 'obc_affine', got '${OBC_ARM}'"; exit 2 ;;
esac
export OBC_ARM

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Do NOT clear CUDA_VISIBLE_DEVICES here: --gres=gpu:1 sets it to the allocated card, and the
# config refuses a device index for exactly that reason. Clearing it makes device='cuda' fail.

# Still needed with a GPU: data loading, the window build, the reference-set build and the eager
# validation free run are all CPU work.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Keep inductor/triton codegen off $HOME (quota) and warm across runs.
# KEYED BY OS IMAGE. /dataB1 is shared across every node and the cluster is heterogeneous,
# so an unkeyed cache lets one node load another's compiled artefacts. Job 85010 died on
# blade2 (Ubuntu 20.04, GLIBC 2.31) importing a __triton_launcher.so that an earlier A100
# job had compiled against GLIBC 2.34:
#   ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found
# Keying on the OS image rather than the hostname keeps same-image nodes sharing a warm
# cache. The first run on each image pays full compilation (about 55 to 61 min of CUDA-graph
# recording before the second update completes; that is NOT a hang, do not cancel it).
CACHE_KEY="$(. /etc/os-release; echo "${ID}${VERSION_ID}")-$(uname -m)"
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache/$CACHE_KEY
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache/$CACHE_KEY
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
echo "cache_key=${CACHE_KEY}"

echo "OBC_ARM=${OBC_ARM}"
echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
date

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true

echo "=== host CPU (t_step tracks HOST dispatch, not the GPU) ==="
lscpu | grep -E "CPU\(s\)|Thread|Core|Socket|Model name"

echo "=== toolchain (inductor needs gcc and triton) ==="
which gcc && gcc --version | head -n 1 || echo "gcc NOT FOUND -> inductor will fail"
python -c "import triton; print('triton', triton.__version__)" || echo "triton NOT FOUND"

# Record the EFFECTIVE imported configuration under this arm, not what the banner intends.
# Deployed copies lag local edits and run 74045's runner echoed a stale config while the job ran
# something else. main() is guarded, so importing runs no training. `|| true` because pipefail.
echo "=== EFFECTIVE imported configuration snapshot (OBC_ARM=${OBC_ARM}) ==="
python -u -c "
import sys; sys.path.insert(0, 'scripts/gantry')
from dataclasses import fields
import gantry_interconnect_dynamic as G
for f in sorted(fields(G.CFG), key=lambda f: f.name):
    print('  %-26s %r' % (f.name, getattr(G.CFG, f.name)))
print('  %-26s %r' % ('DERIVED nf', G.CFG.nf))
print('  %-26s %r' % ('DERIVED na_nb', G.CFG.na_nb))
print('  %-26s %r' % ('DERIVED ts_new', G.CFG.ts_new))
" || true

echo "=== D-192 arm ${OBC_ARM} ==="
srun --cpu-bind=cores python -u scripts/gantry/gantry_interconnect_dynamic.py

date
echo "done ${OBC_ARM}"
