#!/bin/bash
#SBATCH -J final-eval
#SBATCH -p kauai,molokai,lanai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=32gb
#SBATCH -t 24:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/thesis-results-4/%x_%A_%a.out

# Evaluation of the final campaign (THESIS-RESULTS.md section 5, D-269), one array task per model.
# Submit through submit.sh final-eval, which also queues make_results.py after every task.
# The scripts in Thesis-writeup/Code/Evaluation find the runs themselves (eval_common.discover_runs)
# in simulations/gantry_subnet/thesis_taf_lowpass_noise_data_lsq (D-271); THESIS_BB_DIR names the selected
# R8 configuration's output folder. CPU: the scorer runs every model eagerly on the CPU.
#   tasks 0-8  grey box R1 R2 R3 R4 R5 R6 R7 R13 R14: predictions on train, val, test, twins
#   task  9    black box: predictions on train, val, test, twins
#   task  10   overlaps, predicted bias (eval_overlap) and negation (eval_negation), OBC/U pairs

set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 THESIS_EVAL=final
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

EV=Thesis-writeup/Code/Evaluation
TASKS=(R1 R2 R3 R4 R5 R6 R7 R13 R14 BB OBC)
T=${TASKS[${SLURM_ARRAY_TASK_ID:?submit through submit.sh final-eval}]}
echo "task=${T} bb_dir=${THESIS_BB_DIR:-<unset>} node=${SLURM_JOB_NODELIST}"; date
case "$T" in
  BB)  python -u $EV/eval_blackbox.py ;;
  OBC) python -u $EV/eval_overlap.py R4 R1 R5 R2 R14 R13
       python -u $EV/eval_negation.py R4 R1 R14 R13 ;;
  *)   python -u $EV/eval_greybox.py "${T}:physics,base_start,base_true,clamped" --split=train,val,test,twins
       # the frozen-Y LTI baseline of Fig. 1 (step 1) is added to R5's test predictions
       if [ "$T" = R5 ]; then python -u $EV/eval_greybox.py R5 --lti; fi ;;
esac
date; echo "done ${T}"
