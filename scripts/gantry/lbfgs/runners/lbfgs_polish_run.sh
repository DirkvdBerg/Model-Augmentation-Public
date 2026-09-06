#!/bin/bash
#SBATCH -J lbfgs-polish-run
#SBATCH -p hawaii
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 04:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/lbfgs/lbfgs_polish_run%j.out

# THE REAL POLISH RUN (D-171). 500 L-BFGS iterations on the job-81262 checkpoint.
#
# Budget, from the 5-iteration smoke (job 81503, same config otherwise):
#     val_before        ~162 s
#     batch build+warm   ~15 s
#     COMPILE          ~1400 s   one-off, at the phase's own 16384-window shape. Over half the
#                                phase. Silence here is inductor, not a hang.
#     500 iterations   ~500-900 s   (2-6 compiled closures each at ~0.3-0.6 s)
#     val_after         ~162 s
#     phase total       ~40-45 min, then 10-20 min of post-run diagnostics and evaluation.
#
# CFG must read (scripts/gantry/gantry_interconnect_dynamic.py):
#     lbfgs=True            start_phase='lbfgs'
#     lbfgs_max_iter=500    lbfgs_inner_iter=20
#     lbfgs_windows=16384   lbfgs_chunk=None
#     use_f64=False         <- coherent with the float32 checkpoint (D-171)
#     device='cuda'         compile_mode='reduce-overhead'
#     orth=False            joint_estimation=False
# and once, by hand:  mkdir -p ~/logs/augmentation/lbfgs
#
# THE RESULT, and the bar it has to clear. Acceptance is the 12 s free-run sim-RMS over V1-V4,
# and run 81262 was itself oscillating +-0.1% between validations over its last 14. So an
# improvement below ~0.3% is inside the noise Adam was already bouncing around in and is NOT a
# result. The phase rolls back on its own if sim-RMS does not improve at all.
#
# What to read in the output:
#   [lbfgs] determinism check (compiled): ... -> OK    precondition for the line search
#   [lbfgs] iter <=  20 ... 40 ... 60 ...              a line every ~20 iterations
#   [lbfgs] ACCEPTED / ROLLED BACK                     the verdict
#   [lbfgs] sim-RMS 5.774376e-06 -> <after>            the number that decides it
#   [lbfgs] loss <first> -> <last>                     training-loss descent, NOT comparable to
#                                                      Adam's, it is over a different window set
# The `reason` field distinguishes "iteration cap 500 reached" (still descending, consider more)
# from "converged" (the tolerances fired, more iterations would not help).
#
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
