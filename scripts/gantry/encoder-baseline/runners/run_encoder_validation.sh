#!/bin/bash
#SBATCH -J encoder-validation
#SBATCH -p kauai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH -t 14:00:00
#SBATCH --signal=USR1@1800
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/encoder/encoder_validation_%j.out

set -o pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

LOGDIR="/home/dirk_van_den_berg/logs/augmentation/encoder"
mkdir -p "$LOGDIR"
TAG="${SLURM_JOB_ID:-local}"

echo "========================================"
echo "Encoder validation suite  (job $TAG)"
echo "========================================"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
date
echo ""

# Step 0: init diagnostic (fast, no training, ~seconds).
# Encoder standalone baseline: train encoder alone on baseline data (~minutes).
# Encoder standalone MSD: train encoder alone on MSD data (~minutes).
# Steps 1 and 2 pipeline: full SSE_Interconnect training (parallel, ~hours).
# Each step gets its own log file. Failures are isolated.

run_step() {
    local name="$1"
    local script="$2"
    local logfile="${LOGDIR}/encoder_${name}_${TAG}.log"

    echo "  [${name}] start: $(date)"
    if python -u "$script" > "$logfile" 2>&1; then
        echo "  [${name}] status: OK  ($(date))"
    else
        echo "  [${name}] status: FAILED (exit code $?)  ($(date))"
    fi
}

print_summary() {
    local name="$1"
    local logfile="${LOGDIR}/encoder_${name}_${TAG}.log"
    echo "--- ${name} (${logfile}) ---"
    tail -n 20 "$logfile" | sed 's/^/  | /'
    echo ""
}

# --- Step 0: init diagnostic (fast, run first) ---
echo "=== Step 0: init diagnostic ==="
run_step "step0_init" \
    "scripts/gantry/encoder/step0_init_diagnostic.py"
print_summary "step0_init"

# --- Encoder standalone: baseline (should match x_logical perfectly) ---
echo "=== Encoder standalone: baseline ==="
run_step "encoder_baseline" \
    "scripts/gantry/encoder/encoder_baseline_standalone.py"
print_summary "encoder_baseline"

# --- Encoder standalone: MSD (should beat analytical) ---
echo "=== Encoder standalone: MSD ==="
run_step "encoder_msd" \
    "scripts/gantry/encoder/encoder_msd_standalone.py"
print_summary "encoder_msd"

# --- Steps 1 & 2 pipeline: training (run in parallel) ---
echo "=== Steps 1 & 2 pipeline: training (parallel) ==="
run_step "step1_baseline" \
    "scripts/gantry/encoder-baseline/verification/encoder_baseline_full_pipeline.py" &
PID1=$!

run_step "step2_msd" \
    "scripts/gantry/encoder/step2_msd_beat_analytical.py" &
PID2=$!

# Wait for both to finish
wait $PID1
wait $PID2

# --- Summary ---
echo ""
echo "========================================"
echo "All steps finished at $(date)"
echo "========================================"
echo ""
print_summary "encoder_baseline"
print_summary "encoder_msd"
print_summary "step1_baseline"
print_summary "step2_msd"
