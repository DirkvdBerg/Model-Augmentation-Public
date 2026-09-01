#!/bin/bash
#SBATCH -J gantry-gpu-bench2
#SBATCH -p oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH -t 01:30:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/augmentation-closed-loop/gpu_bench2%j.out

# D-169 round 2. Round 1 (job 80606) gave inductor 2.68x -> 240 updates in a 10 h wall at
# nf=12000, against the nf=400 pipeline's 1300. This round chases the two levers that close
# that gap for a SINGLE run:
#   1. `inductor reduce-overhead` (= inductor + CUDA graphs) done properly. Round 1 failed it
#      with "accessing tensor output of CUDAGraphs that has been overwritten", which was a bug
#      in the harness (retained outputs across iterations), not in the model.
#   2. Checkpointing OFF. It costs exactly one extra forward (+33%), and at nf=12000 it is only
#      needed if the graph does not fit. This measures the memory ceiling directly.
#
# -t is 1.5 h rather than 1 h: max-autotune compiles slowly.

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

export BENCH_NF=${BENCH_NF:-200}
export BENCH_BATCHES=${BENCH_BATCHES:-256,512,1024}
export BENCH_CHUNKS=${BENCH_CHUNKS:-0,50}       # 0 = checkpointing OFF
export BENCH_REPS=${BENCH_REPS:-3}
export BENCH_WALL_H=${BENCH_WALL_H:-10}

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
date

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv || true

echo "=== host CPU (t_step tracks HOST dispatch, not the GPU: P2000 and 2080 Ti measured equal) ==="
lscpu | grep -E "Model name|CPU\(s\)|CPU max MHz|Thread|Core|Socket"

echo "=== toolchain ==="
which gcc && gcc --version | head -n 1 || echo "gcc NOT FOUND -> inductor will fail"
python -c "import triton; print('triton', triton.__version__)" || echo "triton NOT FOUND"

echo "=== gantry GPU benchmark round 2 ==="
srun --cpu-bind=cores python -u scripts/gantry/GPU/gpu_bench2.py

date
echo "done"
