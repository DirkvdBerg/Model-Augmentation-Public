#!/bin/bash
# STAGE A. The verification ladder for the fixed-reference trajectory regulariser, on the GPU.
#
# THE QUESTION. V3 to V7 have never completed. There is no measured physical rank, no singular
# spectrum, no conditioning number, no penalty-gradient ownership and no chunking result for the
# gantry, so the Sect. 6.5 rank decision is un-taken and the pilot gate is not passed. This job
# either produces those numbers or names the operation that stops it.
#
# WHY THIS IS NOT JUST "A BIGGER MACHINE". Four local attempts blocked with the process at about
# 1% CPU (16-40 s of CPU per 20 min of wall clock) and the stall point moved between them. The
# cause was never identified: halving the window set, removing a dense N-by-N projector, removing
# a 4800x6774 SVD and pinning threads all failed to move it, and `torch.set_num_threads(1)` made
# it worse. So this run carries a per-operation ledger (wall, CPU, peak RSS, CUDA peak) and a
# watchdog that dumps every thread's stack on an overrun. A fifth stall names its call.
#
# NOTE the local box had 1.6 GB free of 16 GB throughout, which is why this asks for an A100.
#
# BUDGET. Measured locally, CPU, 2 windows: P0/V0/V1/V2 complete in 179 s of wall clock, of which
# V2 is 159 s. V3 has no completed timing at all; the only figure is ~13 s per forward-mode
# physical JVP at 4 windows, and S is 14 of those. The 4 h request is therefore mostly headroom
# for a stage whose cost is genuinely unknown, not an estimate.
#
# PARTITION. lanai/molokai are the A100 nodes (40-80 GB); oahu/mpi are RTX 2080 (8 GB), which is
# not obviously more headroom than the local box had. gpu_bench.sh argues the 2080 wins on clock
# for dispatch-bound work; that is a THROUGHPUT argument and does not apply to a stage we cannot
# complete at all. Revisit once V3 has a measured cost.
#
# Prepare, ONCE, by hand (SLURM resolves -o at schedule time, so the directory must exist first;
# see scripts/gantry/drift-isolation/runners/make_log_dirs.sh for the full reasoning):
#     mkdir -p /home/dirk_van_den_berg/logs/augmentation/orthogonality
# Submit:
#     sbatch scripts/gantry/orthogonality/runners/run_verify_ladder.sh
# Watch:
#     tail -f /home/dirk_van_den_berg/logs/augmentation/orthogonality/verify_ladder_<JOBID>.out
# Interrogate a stall WITHOUT killing it (a stack dump lands in the log):
#     kill -USR1 <pid>
# Results:
#     scripts/gantry/orthogonality/results/<JOBID>/gantry_nominal.json   (+ v3_stacks.log)
#     Read `summary.stages_incomplete` FIRST: a non-empty list means the headline pass count
#     does not cover the stages that were requested. The job exits non-zero in that case.
#
#SBATCH -J orth-ladder
#SBATCH -p lanai,molokai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 04:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/orthogonality/verify_ladder_%j.out

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Do NOT clear CUDA_VISIBLE_DEVICES: --gres=gpu:1 sets it to the allocated card, and verify.py
# refuses to fall back to the CPU rather than silently recording cuda while running elsewhere.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

RUN_DIR=scripts/gantry/meeting/meeting-07-09-2026/server/augmentation_ma50_b140-230_a6_z03_linear_map/81757
OUT_DIR="scripts/gantry/orthogonality/results/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_DIR"

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "run_dir=${RUN_DIR}"
echo "out_dir=${OUT_DIR}"
date --iso-8601=seconds

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

# The generic gates first. They cost seconds, they do not touch the gantry, and a failure here
# invalidates everything below: this is the suite that caught `compressed_nuisance_basis`
# returning an all-zero basis at the right shape and rank.
echo "=== generic gates ==="
srun --cpu-bind=cores python -u scripts/gantry/orthogonality/testbed/test_trajectory_projection.py
srun --cpu-bind=cores python -u scripts/gantry/orthogonality/testbed/test_penalty_hook.py
# The diagnosis tooling itself. If the watchdog does not fire, or the heartbeat is silently
# cancelled by a nested watchdog (it was), a stalled run reports nothing and an empty dump file
# reads as "nothing was wrong".
srun --cpu-bind=cores python -u scripts/gantry/orthogonality/testbed/test_diag.py
# The REAL fit() override, including an optimizer refresh mid-fit. deepSI replaces
# self.optimizer on a resumed system, which silently dropped the penalty wrapper until
# 2026-09-08; this is the check that would catch it coming back.
srun --cpu-bind=cores python -u scripts/gantry/orthogonality/testbed/test_fit_override.py
# The geometry cache key. It shipped keyed on the checkpoint BASENAME, and the content-hash
# fix then missed M_error, so the controller's whole error-feedback path could change without
# changing the key. A stale artifact would project perfectly well against the wrong reference.
srun --cpu-bind=cores python -u scripts/gantry/orthogonality/testbed/test_cache_provenance.py

# --op-timeout 1200: twenty minutes is far longer than any operation should take and far shorter
# than the four attempts that produced nothing. An overrun dumps the stacks and aborts the stage
# with a recorded failure, rather than letting the job sit until the wall clock ends.
# --heartbeat 900: a whole-run stack dump every 15 min, covering the stages the
# per-operation watchdog does not wrap. Across four local stalls the blocked call
# appeared in V1, V2 AND V3, so a watchdog on one stage alone would have missed two
# of them. Costs nothing on a healthy run: faulthandler arms a C-level timer and
# prints only when it fires.
echo "=== ladder: P0/V0/V1/V2 then V3-V7, GPU, 4 windows ==="
srun --cpu-bind=cores python -u scripts/gantry/orthogonality/verify.py \
    --suite gantry \
    --run-dir "$RUN_DIR" \
    --device cuda \
    --windows 4 \
    --stages V3 V4 V6 V7 \
    --op-timeout 1200 \
    --heartbeat 900 \
    --out "$OUT_DIR"

echo ""
echo "Finished at $(date --iso-8601=seconds)"
