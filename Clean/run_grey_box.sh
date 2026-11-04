#!/bin/bash
#SBATCH -J grey-box
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -t 72:00:00
#SBATCH --array=0-8
#SBATCH -o <log folder>/grey_box_%A_%a.out

# Grey-box runs of the thesis: one array index trains one configuration of RUNS on the GPU, then evaluates it.
#   sbatch Clean/run_grey_box.sh                  # all 9
#   sbatch --array=0,1 Clean/run_grey_box.sh      # M_OBC and M_U only
# The OBC polish needs a GPU with about 28 GB of memory.
# Before submitting, fill in the <...> placeholders: partition, log folder, conda environment and repository.

set -eo pipefail
IDX=${SLURM_ARRAY_TASK_ID:?run as an array job}
CONDA_ENV="<conda environment>"   # the environment of environment.yml
REPO="<repository>"               # the folder that contains Clean/

# name and arguments per array index; every other setting is SETTINGS in gantry_interconnect_dynamic.py
RUNS=(
  "M_OBC              --arm obc --start detuned --n-a 2 --ps2"
  "M_U                --arm u   --start detuned --n-a 2 --ps2"
  "obc_correct_na2_ps --arm obc --start correct --n-a 2 --ps2"
  "obc_detuned_na0    --arm obc --start detuned --n-a 0 --no-ps2"
  "u_correct_na2      --arm u   --start correct --n-a 2 --no-ps2"
  "u_correct_na2_ps   --arm u   --start correct --n-a 2 --ps2"
  "u_correct_na0      --arm u   --start correct --n-a 0 --no-ps2"
  "u_correct_na8_ps   --arm u   --start correct --n-a 8 --ps2"
  "u_detuned_na0      --arm u   --start detuned --n-a 0 --no-ps2"
)
read -r NAME ARGS <<< "${RUNS[$IDX]:?no run line for index $IDX}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$REPO"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

echo "index=${IDX} name=${NAME} args=${ARGS} git=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
echo "job_id=${SLURM_JOB_ID} node_list=${SLURM_JOB_NODELIST}"; date
srun --cpu-bind=cores python -u Clean/gantry_interconnect_dynamic.py $ARGS
date
