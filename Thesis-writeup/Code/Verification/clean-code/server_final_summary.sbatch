#!/bin/bash
#SBATCH -J clean-final-summary
#SBATCH -p kauai,molokai,lanai,oahu,hawaii
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=16gb
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/clean-code/%x_%j.out

# D-295 final batch summary: every task's exit code, then each check; writes $WORK/summary.txt (name, PASS/FAIL, the
# check's own line). Bitwise checks of tree B (pinv) are against the first run's mainline outputs (W_OLD).

set -o pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
REPO=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
cd "$REPO/Thesis-writeup/Code/Verification/clean-code"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${WORK:?submit through server_final.sh}"
source "$WORK/settings.sh"
R=$WORK/reports
SUM=$WORK/summary.txt
: > "$SUM"
row() { printf '%-26s %-6s %s\n' "$1" "$2" "$3" >> "$SUM"; }
check() {   # check NAME PATTERN command...: PASS on exit 0; the row keeps the last line matching PATTERN
  local name=$1 pat=$2; shift 2
  "$@" > "$R/$name.log" 2>&1
  local rc=$?
  row "$name" "$([ $rc = 0 ] && echo PASS || echo FAIL)" "$(grep -E "$pat" "$R/$name.log" | tail -1 | cut -c1-300)"
}

# 1. Every task finished with exit code 0
for T in cpu big gpu; do
  while IFS=$'\t' read -r NAME TREE CMD; do
    L=$(grep -h "^TASK_DONE $NAME rc=" "$WORK/logs/$NAME.log" 2>/dev/null | tail -1)
    row "task $NAME" "$([ "$L" = "TASK_DONE $NAME rc=0" ] && echo PASS || echo FAIL)" "${L:-no TASK_DONE line}"
  done < "$WORK/tasks-$T.tsv"
done

# 2. Tree B against mainline, bitwise: overlap, bias, negation and the OBC basis; polish R13; Adam R1 from zero moments
for X in R1 R4 R13; do
  check "l4b-$X" COMPARE_L4B python -u compare_l4b.py "$W_OLD/l4A/mainline-merged" "$WORK/l4P/$X" "$X"
done
export HARNESS_CLEAN_ROOT="$COPY-pinv" EXPECT_TREE=mixed     # tree B is neither D-294 state of d294_revert.py
check polish-R13 COMPARE_PHASE python -u compare_phase.py bitwise "$W_OLD/polish/A-mainline-R13" "$WORK/polish/P-clean-R13" "$R/polish-R13.json"
check adam0-R1 COMPARE_PHASE python -u compare_phase.py bitwise "$W_OLD/adam/A-mainline-R1" "$WORK/adam0/P-clean-R1" "$R/adam0-R1.json"

# 3. Fresh training from scratch, CPU, bitwise (layer 3: both pairs, Adam state, losses, val_hist, config)
HARNESS_GOLDEN="$WORK/l3" HARNESS_CLEAN_OUT="$WORK/l3/_clean" check l3 'COMPARE|PASS|FAIL' python -u compare.py l3 221-lowpass 226-lowpass

# 4. Compiled GPU smokes on tree B against mainline's (first run): training, losses, compiled polish
check gpu COMPARE_GPU python -u compare_gpu.py "$WORK/gpu"
unset EXPECT_TREE

# 5. Clean/'s own evaluation against the report CSV (final tree)
for X in R1 R6; do
  row "eval-$X" "$(grep -q "^EVAL_CHECK $X PASS" "$WORK/logs/eval-$X.log" && echo PASS || echo FAIL)" \
      "$(grep -h '^EVAL_CHECK' "$WORK/logs/eval-$X.log" | cut -c1-300)"
done

# 6. Black box: mainline's eval_blackbox and Clean's eval_bb on the same node, bitwise
python -u compare_bb_preds.py "$WORK/bb-main" "$WORK/bb-clean" > "$R/bb.log" 2>&1
row bb "$(grep -q 'BITWISE IDENTICAL' "$R/bb.log" && echo PASS || echo FAIL)" "$(tail -1 "$R/bb.log")"

# 7. Adam R4, R6 from zero moments on the final tree (server_adam0.sbatch of the first run), if they ran
for X in R4 R6; do
  L=$(grep -h COMPARE_PHASE "$W_OLD/logs/adam0-B-clean-$X.log" 2>/dev/null | tail -1)
  row "adam0-final-$X" "$(echo "$L" | grep -q ' PASS' && echo PASS || echo FAIL)" "${L:-not run}"
done

N_FAIL=$(grep -c ' FAIL ' "$SUM")
echo "FINAL_PROOF $([ "$N_FAIL" = 0 ] && echo PASS || echo "FAIL ($N_FAIL rows)")" >> "$SUM"
cat "$SUM"
