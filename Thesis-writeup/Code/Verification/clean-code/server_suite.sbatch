#!/bin/bash
#SBATCH -J clean-suite
#SBATCH -p kauai,molokai,lanai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=32gb
#SBATCH -t 48:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/clean-code/%x_%A.out

# The local layers of the clean-code harness on the server (D-284 amendment 10, step 5): layers 1 and 2 (16
# cases), 3 (5), 3b (2) and 3c (6 with goldens, 3 clean-only), mainline side first and then the clean side, each
# layer's comparison; the launch smoke with its val check; the branch coverage of the scoreboard (the 16 cases and
# smoke-R1-lowpass); r1_check.py on the staged R1 folder. The goldens in git were captured on Windows and the
# comparators refuse goldens of another data folder or library set, so both sides are captured afresh here, in
# HARNESS_GOLDEN. One thread (bitwise comparison). Run it with the MM16 line REVERTED (step 2 of the commands) and
# after l4_stage.py, which makes STAGING. A failing step is logged (SEQ_END rc) and the job goes on.

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
REPO=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
cd "$REPO/Thesis-writeup/Code/Verification/clean-code"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HARNESS_GOLDEN="${HARNESS_GOLDEN:-$REPO/simulations/clean-golden/server-D284}"
STAGING="${STAGING:-$REPO/simulations/clean-golden/L4-staging-D284}"
LABEL="${LABEL:-residue-after}"
if [ -e "$HARNESS_GOLDEN" ]; then echo "$HARNESS_GOLDEN exists; goldens are never overwritten"; exit 1; fi
echo "golden root ${HARNESS_GOLDEN} staging ${STAGING} label ${LABEL} node ${SLURM_JOB_NODELIST}"; date
sha256sum ../../../../Clean/gantry_dynamic/data.py

run() {
  echo "SEQ_START $*"
  T0=$(date +%s)
  python -u "$@" 2>&1 | grep --line-buffered -av "FutureWarning\|torch.load(\|Gym has been\|gymnasium\|migration guide\|Please upgrade\|Users of this version"
  echo "SEQ_END rc=${PIPESTATUS[0]} seconds=$(( $(date +%s) - T0 )) $*"
}

CASES="221-lowpass 222-lowpass 223-lowpass 224-lowpass 226-lowpass 227-lowpass 229-lowpass 230-lowpass
       221-noisefree 222-noisefree 223-noisefree 224-noisefree 226-noisefree 227-noisefree 229-noisefree 230-noisefree"
for SIDE in mainline clean; do
  for c in $CASES; do run capture_l12.py --side $SIDE --row ${c%%-*} --noise ${c#*-}; done
  run run_l3.py --side $SIDE --row 221 --noise lowpass
  run run_l3.py --side $SIDE --row 229 --noise noisefree
  run run_l3.py --side $SIDE --row 224 --noise noisefree
  run run_l3.py --side $SIDE --row 226 --noise lowpass
  run run_l3.py --side $SIDE --row 227 --noise lowpass
  run run_l3bc.py --side $SIDE --layer 3b --row 221 --noise lowpass
  run run_l3bc.py --side $SIDE --layer 3b --row 226 --noise lowpass
  for k in i ii iii iv; do run run_l3bc.py --side $SIDE --layer 3c --row 226 --noise lowpass --kind $k; done
  for k in i ii; do run run_l3bc.py --side $SIDE --layer 3c --row 221 --noise lowpass --kind $k; done
done
run run_l3bc.py --side clean --layer 3c --row 226 --noise lowpass --kind v
run run_l3bc.py --side clean --layer 3c --row 224 --noise noisefree --kind vi
run run_l3bc.py --side clean --layer 3c --row 226 --noise lowpass --kind vii
for layer in l12 l3 l3b l3c; do run compare.py $layer; done

# the launch smoke (clean-up-verification/smoke.py defaults) and its val check
run ../../../../clean-up-verification/smoke.py | tee smoke-suite.log
SMOKE_RUN_DIR=$(grep "^SMOKE_RUN_DIR" smoke-suite.log | tail -1 | cut -d' ' -f2-)
run ../../../../clean-up-verification/val_check.py "$SMOKE_RUN_DIR"

# branch coverage of the grey box over the 16 cases and smoke-R1-lowpass (scoreboard.py, step 4b method)
run scoreboard.py coverage --label "$LABEL"
run scoreboard.py report --label "$LABEL"

# evaluate() on the converted thesis R1 folder (job 89792) against res_errors.csv, MM16 reverted
run ../../../../clean-up-verification/r1_check.py "$STAGING/89792"
date; echo "CLEAN_SUITE_SERVER_DONE"
