#!/bin/bash
#SBATCH -J gantry-gpu-bench
#SBATCH -p oahu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH -t 01:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/augmentation-closed-loop/gpu_bench%j.out

# D-169 GPU benchmark. Measures t_step, its (non-)dependence on batch size, and whether the
# `inductor` backend fuses the op count down. inductor is the ONLY torch.compile backend that
# generates fused kernels, and it cannot run on the Windows development PC (needs MSVC; Triton
# has no Windows support) -- which is the entire reason this job exists.
#
# Partition `oahu` (blade1/blade2, RTX 2080) NOT `lanai`/`molokai` (A100): the workload is
# dispatch-bound, so per-kernel latency tracks CLOCK, where the 2080 (~1.8 GHz) leads the A100
# (~1.41 GHz). The A100's bandwidth and VRAM address a bottleneck this model does not have.
#
# -c 4 is deliberate. The rollout is ONE Python thread issuing ~254 ops per timestep; extra cores
# do not shorten it. The 4 are for data loading.

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject

cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

export PYTHONUNBUFFERED=1

# NOTE: CUDA_VISIBLE_DEVICES is deliberately NOT cleared here (the CPU runner sets it to "").
# SLURM's --gres=gpu:1 sets it to the allocated card, which is what makes device='cuda' correct
# without ever naming an index.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# inductor COMPILES the code it generates, so it needs writable cache dirs and a C++ compiler.
# Default is /tmp/torchinductor_$USER, which on a cluster node can be small or purged between
# jobs; pointing both at /dataB1 also makes the compiled kernels reusable across jobs.
export TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache
export TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Benchmark knobs (see gpu_bench.py). t_step is nf-independent, so nf stays small.
export BENCH_NF=${BENCH_NF:-200}
export BENCH_CHUNK=${BENCH_CHUNK:-50}
export BENCH_BATCHES=${BENCH_BATCHES:-256,1024,4096}
export BENCH_REPS=${BENCH_REPS:-2}

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
date

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv || true

echo "=== toolchain (inductor needs a C++ compiler; failure here explains an inductor failure) ==="
which gcc && gcc --version | head -n 1 || echo "gcc NOT FOUND -> inductor will fail"
python -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import triton; print('triton', triton.__version__)" || echo "triton NOT FOUND -> inductor cannot generate GPU kernels"

echo "=== gantry GPU benchmark ==="
srun --cpu-bind=cores python -u scripts/gantry/GPU/gpu_bench.py

date
echo "done"
