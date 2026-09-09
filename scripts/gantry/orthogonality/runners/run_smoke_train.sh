#!/bin/bash
# STAGE B. A hard-capped paired smoke run: does the penalty actually reach the optimizer?
#
# THE QUESTION. The trajectory penalty is a gradient contribution, not a loss term, because the
# production closure is `loss -> zero_grad -> backward` and anything backwarded inside `loss()`
# is wiped a moment later. `SSE_Interconnect_Composed.fit` wraps `optimizer.step` so the
# accumulation lands between the closure returning and the update. This job runs a few real
# updates through that path and checks the hook fires once per update, the gradient lands in
# `eta` and nowhere else, and a paired `traj_orth=False` arm is unaffected.
#
# THIS IS NOT AN EFFICACY RUN. No beta sweep, no recovery study, no held-out comparison, and the
# script refuses `--its > 50` for exactly that reason. `traj_beta` is a CHOSEN CONSTANT WITH NO
# JUSTIFICATION until an L-curve is run (RULES.md, 2026-09-07), and this job does not select it.
#
# GATED ON STAGE A. The geometry this uses is the one the ladder builds, and the ladder's V3-V7
# have never completed. The guard below refuses to start unless stage A's JSON reports every
# requested stage complete: a run on a collapsed or unmeasured geometry would look like a run and
# mean nothing.
#
# Prepare, ONCE, by hand:
#     mkdir -p /home/dirk_van_den_berg/logs/augmentation/orthogonality
# Submit, AFTER stage A has finished cleanly (chain it if you prefer):
#     LADDER_JOB=<stage A job id> sbatch scripts/gantry/orthogonality/runners/run_smoke_train.sh
#     # or:  sbatch --dependency=afterok:<stage A job id> ... LADDER_JOB=<id>
# Watch:
#     tail -f /home/dirk_van_den_berg/logs/augmentation/orthogonality/smoke_train_<JOBID>.out
# Results:
#     scripts/gantry/orthogonality/results/smoke_<JOBID>/smoke_train.json
#
#SBATCH -J orth-smoke
#SBATCH -p lanai,molokai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 02:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/orthogonality/smoke_train_%j.out

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Do NOT clear CUDA_VISIBLE_DEVICES: --gres=gpu:1 sets it to the allocated card.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

RUN_DIR=scripts/gantry/meeting/meeting-07-09-2026/server/augmentation_ma50_b140-230_a6_z03_linear_map/81757

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "ladder_job=${LADDER_JOB:-<unset>}"
date --iso-8601=seconds

# ---- THE GATE. Refuse to run on an unverified geometry. -----------------------------------
LADDER_JSON="scripts/gantry/orthogonality/results/${LADDER_JOB}/gantry_nominal.json"
if [[ -z "${LADDER_JOB:-}" || ! -s "$LADDER_JSON" ]]; then
  echo "LADDER_JOB is unset or $LADDER_JSON is missing. Stage B runs on the geometry stage A" >&2
  echo "builds and verifies; without that record there is nothing to certify it." >&2
  exit 2
fi
python - "$LADDER_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
s = d.get('summary', {})
inc, failed = s.get('stages_incomplete') or [], s.get('failed', 0)
print(f"stage A: passed={s.get('passed')} failed={failed} incomplete={inc}")
if failed or inc:
    sys.exit("stage A did not complete cleanly; stage B would be training against a geometry "
             "whose rank and conditioning were never measured. Fix stage A first.")
r3 = (d.get('V3') or {})
print(f"stage A geometry: rank(S)={r3.get('rank_S_raw')} -> rank(Sbar)={r3.get('rank_S_profiled')}"
      f"  rank(Q_E)={r3.get('rank_E')}  N={r3.get('S_shape', [None])[0]}")
PY

echo "=== generic gates ==="
srun --cpu-bind=cores python -u scripts/gantry/orthogonality/testbed/test_penalty_hook.py
srun --cpu-bind=cores python -u scripts/gantry/orthogonality/testbed/test_fit_override.py

echo "=== paired smoke: 3 capped updates, penalty on vs off ==="
srun --cpu-bind=cores python -u scripts/gantry/orthogonality/smoke_train.py \
    --run-dir "$RUN_DIR" \
    --device cuda \
    --its 3 \
    --windows 4 \
    --beta 1.0

echo ""
echo "Finished at $(date --iso-8601=seconds)"
