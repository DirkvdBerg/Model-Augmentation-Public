#!/bin/bash
#SBATCH -J lbfgs-polish-smoke
#SBATCH -p hawaii
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 02:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/lbfgs/lbfgs_polish_smoke%j.out

# PLUMBING TEST OF THE REAL PATH, capped at 5 L-BFGS iterations. Not a result.
#
# This runs the PRODUCTION entry point, not a probe, so it exercises everything the full run
# will: resume from a checkpoint, architecture inferred from it, the dtype-coherence guard, the
# eager warm-up, compilation at the phase's own batch shape, the determinism check, real L-BFGS
# steps, both validations, the acceptance decision, and the checkpoint writes with job-id naming
# and the optimizer-drop rule.
#
# It also answers the three things the 81463 probe could not, because that swept EAGER:
#   * does 16384 windows fit COMPILED (CUDA graphs reserve their own pools)?
#   * what does a compiled closure actually cost?
#   * what does the one-off compilation at this new shape cost? (~500 s expected: fit() compiled
#     for batch 512 and the phase uses 16384, so inductor recompiles. Not a hang.)
#
# BEFORE SUBMITTING, set in gantry_interconnect_dynamic.py's CFG:
#     lbfgs=True                      start_phase='lbfgs'
#     lbfgs_max_iter=5                lbfgs_inner_iter=1        <- THE CAP, this is a smoke test
#     lbfgs_windows=16384             lbfgs_chunk=None
#     use_f64=False                   <- coherent with the float32 checkpoint (D-171)
#     device='cuda'                   compile_mode='reduce-overhead'
#     orth=False                      joint_estimation=False
# and once, by hand:  mkdir -p ~/logs/augmentation/lbfgs
#
# WHAT TO CHECK IN THE OUTPUT
#   [closed loop] training rollout COMPILED            compilation engaged
#   [lbfgs] fixed batch: 16384 of 66612 windows (stride 10 -> 40)
#                                                      the stride scaling worked
#   [lbfgs] determinism check (compiled): ... -> OK    the line search precondition holds
#   [lbfgs] iter <=   1 ... ( N closures, T s)         per-iteration cost, COMPILED this time
#   [lbfgs] ACCEPTED / ROLLED BACK                     either is a pass for a smoke test
#   sim-RMS <before> -> <after>                        both measured in one code path
#   Adam state omitted: no Adam phase ran in this job  the optimizer-drop rule fired
#   Checkpoint weights: .../gantry_ckpt_<jobid>.pt     job-id naming, not a random code
#
# A ROLLED BACK result here means nothing about the method: 5 iterations is not a polish. The
# question this job answers is whether the machinery runs, not whether it helps.
#
# THEN the real run: same config with lbfgs_max_iter=500 and lbfgs_inner_iter=20. The bar is a
# sim-RMS improvement beyond run 81262's own +-0.1% validation oscillation, so about 0.3%.

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1

# Do NOT clear CUDA_VISIBLE_DEVICES: --gres=gpu:1 sets it to the allocated card and the config
# refuses a device index for exactly that reason.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# The checkpoint to polish: job 81262, the best of the three in meeting-07-09-2026
# (sim-RMS 5.774376e-06, against 5.956e-06 and 7.273e-06). Float32, nx_ann=8, 24x3; the
# architecture is read off the file and the dtype guard refuses a mismatched run.
# The `_best` suffix is optional, resolve_checkpoint accepts the full .pth path too.
export RESUME_CHECKPOINT=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation/scripts/gantry/meeting/meeting-07-09-2026/server/checkpoints/SSE_Interconnect_Composed_p2LPDA_best

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "resume_checkpoint=${RESUME_CHECKPOINT}"
date

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "=== unit gates (fast, and a failure here invalidates everything below) ==="
srun --cpu-bind=cores python -u -m unittest discover \
    -s scripts/gantry/gantry_dynamic/tests -p "test_lbfgs_polish.py"

echo "=== capped polish on the real entry point ==="
srun --cpu-bind=cores python -u scripts/gantry/gantry_interconnect_dynamic.py

date
echo "done"
