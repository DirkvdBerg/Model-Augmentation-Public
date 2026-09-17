#!/bin/bash
#SBATCH -J obc-probe
#SBATCH -p molokai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 02:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/orthogonal-by-construction/obc_probe%j.out

# PREFLIGHT for the D-192 projection pair. Runs NO training, writes no checkpoint, about ten
# minutes. Three steps, in order:
#
#   0. experiment_configs.py  proves what OBC_ARM changes: the unset default is untouched, and
#      the two arms differ in exactly one field.
#   1. probe_prior_scale.py   (D-193) the D-076 parameter prior against the data term along the
#      recovery path. Measured locally at 49x to 1433x in the prior's favour, which is why the
#      arms run with param_prior=False; this confirms it on the real card.
#   2. probe_compile.py       (D-192 sect. 9.7, D-194) does the corrected step compile without
#      per-objective recompilation, and does one PRODUCTION-SHAPE update (nf 400, batch 512, full
#      reference set) fit on the card. This is THE open question: the correction now uses the
#      hand-written forward sensitivity instead of `torch.func.jvp`, so there is no functorch in
#      the step for inductor to miscompile. Local CPU timing predicts the correction roughly
#      doubles a rollout step (2.07x) rather than the 10.4x the jvp cost.
#
# Submit:
#     sbatch scripts/gantry/orthogonal-by-construction/runners/run_obc_probe.sh
# Read:
#     grep -A3 "==== verdict"        <log>     # the prior question
#     grep -A4 "\[rung 1 contract\]" <log>     # does it compile, does it recompile
#     grep -A6 "\[memory\]"          <log>     # does the production shape fit
# JSON also lands under implementation/results/<date>/.
#
# NO --signal=USR1@1800 (the only line dropped from the gantry_interconnect_dynamic_gpu.sh
# template): nothing installs a SIGUSR1 handler and Python's default action for it is to
# terminate, so on this 1 h allocation the job would be killed at 30 minutes.

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Do NOT clear CUDA_VISIBLE_DEVICES here: --gres=gpu:1 sets it to the allocated card, and the
# config refuses a device index for exactly that reason. Clearing it makes device='cuda' fail.

# Still needed with a GPU: data loading, the window build and the reference-set build are CPU work.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Keep inductor/triton codegen off $HOME (quota) and warm for the training runs that follow.
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Each probe picks its own configuration; an inherited arm would silently change what they measure.
unset OBC_ARM

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
date

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true

echo "=== host CPU (t_step tracks HOST dispatch, not the GPU) ==="
lscpu | grep -E "CPU\(s\)|Thread|Core|Socket|Model name"

echo "=== toolchain (inductor needs gcc and triton; capability must be >= 7.0) ==="
which gcc && gcc --version | head -n 1 || echo "gcc NOT FOUND -> inductor will fail"
python -c "import triton; print('triton', triton.__version__)" || echo "triton NOT FOUND"

echo "=== 0. arm wiring: what OBC_ARM actually changes ==="
srun --cpu-bind=cores python -u \
    scripts/gantry/orthogonal-by-construction/implementation/08-one-step/experiment_configs.py

echo "=== 1. parameter prior scale (D-193) ==="
srun --cpu-bind=cores python -u \
    scripts/gantry/orthogonal-by-construction/implementation/08-one-step/probe_prior_scale.py

# The ladder, one rung per PROCESS. Job 83796 died with an illegal memory access inside
# cudagraph_trees.py, which poisons the CUDA context, so a rung that crashes must not take the
# rest of the probe with it. `|| true` lets each rung fail on its own and the walk continue;
# each writes results/<date>/probe_compile_<mode>.json and says OK or FAILED in the log.
echo "=== 2. compile ladder (D-192 sect. 9.7), one rung per process ==="
# CHANGED (D-194): the step no longer contains a functorch transform. `GantryOBCCorrection` now
# uses the hand-written forward sensitivity, validated against `torch.func.jvp` per term and end
# to end in test_tangent.py, so the illegal memory access that killed jobs 83796 and 83913 has no
# code path left to come from. That puts the PRODUCTION modes back in play, and they are what
# this ladder now walks; the `-nofullgraph` rungs existed only to route around the jvp and are
# not worth a slot. `eager` stays last as the reference.
# CUDA GRAPHS ARE NOT BEING GIVEN UP. Job 83948 showed `reduce-overhead` spending 623.5 s on one
# update at nf 400 and I first read that as a per-update cost. It is the ONE-OFF CUDA-graph
# recording over 400 sequential invocations, which gantry_interconnect_dynamic_gpu.sh already
# documents as "~500 s of ONE-OFF compile on the first two updates", after which job 83795 runs at
# 0.77 s. `default` shows the same shape harmlessly: 14.82 s first, 0.97 s steady. So
# `reduce-overhead` goes FIRST and the wall limit is 2 h, giving it room to reach steady state;
# the probe now judges the SECOND production update, which is what separates a one-off from a
# real per-update cost. At nf 20 on this card it was already the faster rung, 0.08 s to 0.11 s.
for MODE in reduce-overhead default eager; do
    echo ""
    echo "----- rung: ${MODE} -----"
    srun --cpu-bind=cores python -u \
        scripts/gantry/orthogonal-by-construction/implementation/08-one-step/probe_compile.py \
        --mode "${MODE}" || echo "rung ${MODE} FAILED (see above); continuing"
done

echo ""
echo "=== ladder summary ==="
python -u -c "
import glob, json, os
rows = []
for p in sorted(glob.glob('scripts/gantry/orthogonal-by-construction/implementation/results/*/probe_compile_*.json')):
    d = json.load(open(p))
    m = d.get('production_memory', {})
    on, off = m.get('on') or {}, m.get('off') or {}
    rows.append((d.get('mode'), d.get('ok'), off.get('seconds'), on.get('seconds'),
                 m.get('hours_per_100_epochs_on'), on.get('peak_bytes'), d.get('error', '')[:70]))
hdr = ('mode', 'ok', 'nf400 off s', 'nf400 on s', 'h/100ep', 'peak GB', 'error')
print('%-28s %-6s %11s %11s %9s %8s  %s' % hdr)
for mo, ok, so, sn, hr, pk, err in rows:
    f = lambda v, p='%.2f': (p % v) if v else '-'
    print('%-28s %-6s %11s %11s %9s %8s  %s'
          % (mo, ok, f(so), f(sn), f(hr, '%.1f'), f(pk/1e9) if pk else '-', err))
print()
print('Reference: the compiled control arm (83795, no OBC) runs at 0.77 s/update,')
print('           i.e. 2.8 h per 100 epochs. Eager with OBC was 30.47 s, i.e. 110 h.')
ok_rows = [(mo, sn) for mo, ok, so, sn, hr, pk, err in rows if ok and sn and mo != 'eager']
if ok_rows:
    best = min(ok_rows, key=lambda r: r[1])
    print('FASTEST WORKING RUNG: %s at %.2f s/update (%.1f h per 100 epochs).'
          % (best[0], best[1], best[1] * 130 * 100 / 3600.0))
else:
    print('NO compiled rung survived; only eager works, which is too slow for a full run.')
" || true

date
echo "done"
