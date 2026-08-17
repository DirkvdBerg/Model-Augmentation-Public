#!/bin/bash
# STEP 6: closed-loop training, controller IN THE MODEL, ANN routed to ALL EIGHT states, 12 epochs.
# Hypothesis, pre-registered falsifiers and known confounds: docs/gantry-augmentation-problem-log.md
# Section 12, row "STEP 6: CONTROLLER IN THE MODEL". Decisions: D-140 (placement), D-141 (4 kHz,
# location, per-record Cfb), D-142 (xc = 0 is Kessels Remark 5.4; 77x headroom).
#
# LEARNING RATE: 1e-7, NOT the 1e-3/1e-4 used elsewhere.
#   ann_route_ix = (0..7) includes the K = 0 rows (X/Y: 0,2,3,5), and config.py:62 states those
#   need "a much smaller lr (~1e-7)" per D-101/D-102. A first launch at 1e-3 hit a NaN in the
#   training loss at iteration 81. lr is passed via cfg so build_model gives it to init_model
#   (D-101); the script asserts the optimizer actually carries it rather than overriding it.
#
# EPOCHS, not timeout. deepSI cannot honour both: with `timeout` set it runs itertools.count() and
# IGNORES epochs (interconnect.py:604). CL_TIMEOUT is therefore left unset and -t is the only
# backstop, sized with ~2x margin:
#     12 epochs x 260 its/epoch          = 3120 iterations
#     3120 x 5.15 s/it (LOCAL measured)  = 4.5 h training
#     12 validations x ~8 min            = 1.6 h   (full closed-loop free run, all 4 val records)
#     untrained + 2 final evaluations    = 0.4 h
#                                        = 6.5 h expected, -t 14:00:00
#   Measured locally at nf=400, batch 256, stride 10; there is no measurement for kauai, hence the
#   margin. On a hard kill deepSI's _best/_last checkpoints survive but the result JSON and the
#   PER-RECORD attribution do not, because they run after fit() returns (lesson from run 74045).
#
# NO --signal=USR1@1800, deliberately, and this differs from the runner this was based on.
#   Nothing in scripts/gantry/ installs a SIGUSR1 handler (checked), and Python's default action
#   for SIGUSR1 is to TERMINATE. Sending it 30 min before the wall clock would kill the job early
#   and lose exactly the post-fit artifacts the margin above exists to protect. run_stage1.sh
#   documents the same conclusion. If a graceful-stop signal is wanted, the handler has to be added
#   to the Python script first.
#
# Submit:  sbatch scripts/gantry/closed-loop-controller/runners/run_step6.sh
# Watch:   tail -f ~/logs/augmentation/closed-loop-controller/step6_<JOBID>.out
# Result:  scripts/gantry/closed-loop-controller/runs/step6_result_<JOBID>.json
#SBATCH -J cl-step6-allstates
#SBATCH -p kauai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=64gb
#SBATCH -t 14:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/closed-loop-controller/step6_%j.out

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Force CPU usage.
export CUDA_VISIBLE_DEVICES=""

# Use the CPUs allocated by SLURM.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── run configuration ────────────────────────────────────────────────────────
export CL_EPOCHS=12
export CL_LR=1e-7          # K=0 rows are in the route; config.py:62, D-101/D-102. 1e-3 NaN-ed.
export CL_STRIDE=10
export CL_ITS_PER_VAL=epoch
# CL_TIMEOUT deliberately UNSET: setting it would make deepSI ignore CL_EPOCHS entirely.
# CL_TAG defaults to SLURM_JOB_ID inside the script, so results cannot overwrite each other.

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "CL_EPOCHS=${CL_EPOCHS}  CL_LR=${CL_LR}  CL_STRIDE=${CL_STRIDE}  CL_ITS_PER_VAL=${CL_ITS_PER_VAL}"
date

echo "=== CPU info ==="
lscpu | grep -E "CPU\(s\)|Thread|Core|Socket|Model name"

# Record what actually ran rather than what this banner intended: deployed copies lag local edits,
# and run 74045's runner echoed a stale config while the job ran something else.
echo "=== script defaults (env above overrides these) ==="
grep -n "^ROUTE = \|^EPOCHS = \|^LR = \|^STRIDE = \|^N_ITS = " \
    scripts/gantry/closed-loop-controller/cl_step6_run.py

echo "=== step 6: controller in the model, ANN -> all eight states ==="
srun --cpu-bind=cores python -u scripts/gantry/closed-loop-controller/cl_step6_run.py

echo ""
echo "========================================"
echo "Finished at $(date)"
echo "========================================"
