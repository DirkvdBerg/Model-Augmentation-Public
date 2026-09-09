#!/bin/bash
# STAGE C. GPU + torch.compile parity ladder for the trajectory-orthogonality penalty, then the
# longer run gated behind it.
#
# THE QUESTION. Every check so far ran eager on CPU. Production trains on CUDA with
# torch.compile (RunConfig.compile_mode, wired at gantry_dynamic/controller.py:257, measured
# about 6.5x with 'reduce-overhead' in jobs 80610/80634/80652). Does the compiled path execute
# the SAME mathematics? Completing a long run proves nothing on its own.
#
# WHY THIS PENALTY IS UNUSUALLY EXPOSED TO COMPILATION. The intervention is a MUTATED BUFFER:
# ann_gate swaps Static_ANN_Block.out_gate between the full / off / clamped rollouts. Dynamo
# guards on module attributes, so a gate switch either recompiles (slow, correct) or, if the
# value is specialised into the graph, is IGNORED (fast, and silently wrong). A specialised gate
# does not crash and does not zero the penalty, because the latent-init half of the intervention
# still differs; it just makes the penalty a different, unspecified quantity. Stage 2 probes the
# gate directly with the clamped rollout, which differs from full ONLY by the gate.
#
# TOLERANCES ARE MEASURED, NOT CHOSEN. Compiled kernels reassociate sums, so bit-exactness is the
# wrong acceptance criterion and demanding it would only invite loosening it later until it
# passed. Stage 1 runs the identical eager computation TWICE and takes the run-to-run difference
# (atomics, cuDNN selection, reduction order) as the floor this machine can distinguish from
# zero. Stage 2 accepts a compiled quantity within 10x that measured floor. The factor 10 is a
# HEURISTIC margin, recorded as one in the JSON.
#
# THE LADDER, each stage gating the next:
#     0  preflight: CUDA present, compute capability >= 7.0, inductor importable
#     1  eager reference, run twice, floor measured
#     2  compiled parity: p_full, p_off, p_clamped, the gate, V, per-group gradients, one update
#     3  save / load / resume under compilation
#     4  the longer run, ONLY if 0-3 all passed
#
# Stage 4 REFUSES to start if any earlier check failed, and runs eager (saying so) if the
# compiled arm never ran. A skipped parity stage is not a passed one.
#
# NOT AN EFFICACY EXPERIMENT. beta is fixed at 1, a chosen constant with no justification, and
# stage 4 has no control arm. This measures whether the penalty RUNS correctly and what it costs,
# never whether it helps.
#
# PARTITION: hawaii, matching gantry_interconnect_dynamic_gpu.sh. That is the partition where
# compilation is DEMONSTRATED to work end to end (job 80713 on an RTX 2080 Ti: 0.50 s/update
# compiled against 5.0 s/update on the CPU sweep 80498-80557), and where the toolchain inductor
# needs is known to be present. An A100 partition was the first choice here for memory headroom,
# which is the wrong criterion: stage C is a small parity probe, not the stalling geometry build,
# and running it on different hardware from the production training runs would make the timing
# incomparable with the only measurements that exist.
#
# Prepare, ONCE, by hand (SLURM resolves -o at schedule time, so the directory must exist first):
#     mkdir -p /home/dirk_van_den_berg/logs/augmentation/orthogonality
# Submit:
#     sbatch scripts/gantry/orthogonality/runners/run_gpu_compile.sh
# Watch:
#     tail -f /home/dirk_van_den_berg/logs/augmentation/orthogonality/gpu_compile_<JOBID>.out
# Results:
#     scripts/gantry/orthogonality/results/gpu_ladder_<JOBID>/gpu_ladder.json
#     Read `summary.failures` FIRST, then `stages.2_compiled.out_of_tolerance` (which quantities
#     exceeded the measured floor) and `stages.2_compiled.speedup_vs_eager`. If
#     `stages.2_compiled.skipped` is set, stage 4 ran EAGER or not at all.
#
# SANITY LINES TO GREP, the same three the production GPU runner lists:
#   "training rollout COMPILED"          compilation engaged (printed by controller.py)
#   NO "skipping cudagraphs due to cpu"  mode='reduce-overhead' kept its CUDA-graph fast path.
#                                        If this appears, the run silently proceeds at
#                                        inductor-only speed (~2.6x instead of ~6.5x, jobs 80610
#                                        and 80634) and the stage 2 speedup figure is not the
#                                        production one.
#   "the compiled arm really holds compiled callables"   stage 2 asserts this against the object
#
# ONE TRAP WORTH KNOWING BEFORE READING THE OUTPUT. `ClosedLoopSimulator._rollout` selects the
# compiled callable only when `ufuture.is_cuda AND torch.is_grad_enabled()` (closed_loop.py:554).
# A rollout under `torch.no_grad()` therefore runs EAGER even here. That is deliberate in
# production (it keeps validation and the ~20 diagnostics off the compiled artefact), but it
# means the penalty's `d` is a COMPILED p_full minus an EAGER p_off. Stage 2 measures that
# mixed-path term as a fraction of |d| rather than assuming it negligible.
#
# ITERATING ON CRASHES. This is expected to fail a few times before it works. Compilation
# failures surface as a traceback inside inductor or dynamo; the JSON is written incrementally
# after every stage, so a crash in stage 2 still leaves stages 0 and 1 on disk. To retry only the
# cheap part while fixing a codegen error, set STAGES=0,1 below. To confirm the mathematics is
# fine and the problem is purely compilation, run with --no-compile: it takes the eager fallback
# WITHOUT recording stage 2 as passed.
#
#SBATCH -J orth-gpu-compile
#SBATCH -p hawaii
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 03:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/orthogonality/gpu_compile_%j.out

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Do NOT clear CUDA_VISIBLE_DEVICES: --gres=gpu:1 sets it to the allocated card, and stage 0
# fails loudly rather than silently recording cuda while running on the CPU.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Inductor and Triton write generated kernels here. On a shared filesystem the default lands in
# $HOME and collides between concurrent jobs; a cold cache is also why the FIRST compiled call
# takes minutes while later ones take milliseconds, which is why stage 2's timing excludes it.
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Turn a silent eager fallback into a visible one. Dynamo suppresses errors by default and drops
# back to eager, which would let a broken compile report itself as a passing parity check.
export TORCHDYNAMO_VERBOSE=1
export TORCH_LOGS="recompiles,graph_breaks"

RUN_DIR=scripts/gantry/meeting/meeting-07-09-2026/server/augmentation_ma50_b140-230_a6_z03_linear_map/81757
OUT_DIR="scripts/gantry/orthogonality/results/gpu_ladder_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_DIR"

STAGES="${STAGES:-0,1,2,3}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
TRAIN_ITS="${TRAIN_ITS:-200}"
WINDOWS="${WINDOWS:-2}"
BATCH="${BATCH:-8}"

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "run_dir=${RUN_DIR}"
echo "out_dir=${OUT_DIR}"
echo "stages=${STAGES}  compile_mode=${COMPILE_MODE}  train_its=${TRAIN_ITS}"
date --iso-8601=seconds

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true

# Host CPU: the rollout is DISPATCH-BOUND, so stage 2's timing tracks host dispatch more than it
# tracks the GPU. Recorded for the same reason gantry_interconnect_dynamic_gpu.sh records it.
echo "=== host CPU (timing tracks HOST dispatch, not the GPU) ==="
lscpu | grep -E "CPU\(s\)|Thread|Core|Socket|Model name"

# Inductor needs gcc and triton. Checking here fails in one readable line BEFORE Python starts,
# instead of somewhere inside codegen. Stage 0 re-checks from Python; this catches the case where
# the interpreter cannot even reach that point.
echo "=== toolchain (inductor needs gcc and triton) ==="
which gcc && gcc --version | head -n 1 || echo "gcc NOT FOUND -> inductor will fail"
python -c "import triton; print('triton', triton.__version__)" || echo "triton NOT FOUND"

# The generic gates first. Seconds to run, they do not touch the GPU, and a failure here
# invalidates everything below.
echo "=== generic gates ==="
srun --cpu-bind=cores python -u scripts/gantry/orthogonality/testbed/test_trajectory_projection.py
srun --cpu-bind=cores python -u scripts/gantry/orthogonality/testbed/test_penalty_hook.py
srun --cpu-bind=cores python -u scripts/gantry/orthogonality/testbed/test_fit_override.py
srun --cpu-bind=cores python -u scripts/gantry/orthogonality/testbed/test_cache_provenance.py

echo "=== GPU + compile ladder ==="
srun --cpu-bind=cores python -u scripts/gantry/orthogonality/run_gpu_ladder.py \
    --run-dir "$RUN_DIR" \
    --device cuda \
    --compile-mode "$COMPILE_MODE" \
    --stages "$STAGES" \
    --windows "$WINDOWS" \
    --batch "$BATCH" \
    --beta 1.0 \
    --train-its "$TRAIN_ITS" \
    --out "$OUT_DIR"

echo ""
echo "Finished at $(date --iso-8601=seconds)"
