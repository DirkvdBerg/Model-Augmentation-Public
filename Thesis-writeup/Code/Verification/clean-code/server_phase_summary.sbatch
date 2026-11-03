#!/bin/bash
#SBATCH -J clean-phase-summary
#SBATCH -p kauai,molokai,lanai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=16gb
#SBATCH -t 04:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/clean-code/%x_%j.out

# D-295: after every task of server_phase.sh: the task exit codes, the merges and comparisons, coverage. Writes $WORK/summary.txt (one row per check: name, verdict, the check's own line) and the
# reports in $WORK/reports/. The proof holds only when every row says PASS.

set -o pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
REPO=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
cd "$REPO/Thesis-writeup/Code/Verification/clean-code"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${WORK:?submit through server_phase.sh}"
source "$WORK/settings.sh"
R=$WORK/reports
SUM=$WORK/summary.txt
: > "$SUM"
row() { printf '%-34s %-8s %s\n' "$1" "$2" "$3" >> "$SUM"; }
check() {   # check NAME command...: PASS on exit 0; the row keeps the check's last COMPARE/MERGE line
  local name=$1; shift
  "$@" > "$R/$name.log" 2>&1
  local rc=$?
  row "$name" "$([ $rc = 0 ] && echo PASS || echo FAIL)" "$(grep -E '^(COMPARE|MERGE)' "$R/$name.log" | tail -1 | cut -c1-300)"
}

# 1. Every task finished with exit code 0
while IFS=$'\t' read -r NAME ARM COV CMD; do
  L=$(grep -h "^TASK_DONE $NAME rc=" "$WORK/logs/$NAME.log" 2>/dev/null | tail -1)
  row "task $NAME" "$([ "$L" = "TASK_DONE $NAME rc=0" ] && echo PASS || echo FAIL)" "${L:-no TASK_DONE line}"
done < "$WORK/tasks.tsv"

# the arm-A comparators check the clean outputs against this tree (H2)
export HARNESS_CLEAN_ROOT="$COPY-reverted"
check tree-A-manifest bash -c "cd '$COPY-reverted' && sha256sum -c --quiet '$WORK/clean-reverted.sha256'"
check tree-B-manifest bash -c "cd '$COPY-final' && sha256sum -c --quiet '$WORK/clean-final.sha256'"

# 2. Evaluation: merge per side; arm A bitwise; both arms against the report
for S in l4A/mainline l4A/clean l4B/clean; do
  check "merge-${S//\//-}" python -u merge_l4.py "$WORK/$S-merged" $(for X in $L4_RUNS; do echo "$WORK/$S/$X"; done)
done
check l4-A-bitwise python -u compare_l4.py "$WORK/l4A/mainline-merged" "$WORK/l4A/clean-merged"
cp compare-l4.json "$R/compare-l4.json" 2>/dev/null
for S in l4A/mainline l4A/clean l4B/clean; do
  ( check "csv-${S//\//-}" python -u compare_csv.py l4 "$WORK/$S-merged" "$CSV" "$R/csv-${S//\//-}.json" )
done
check csv-bb python -u compare_csv.py bb "$WORK/bb" "$CSV" "$R/csv-bb.json"

# 3. Adam resume: arm A bitwise; arm B against arm A at the noise floor
for X in $ADAM_RUNS; do
  check "adam-A-bitwise-$X" python -u compare_phase.py bitwise "$WORK/adam/A-mainline-$X" "$WORK/adam/A-clean-$X" "$R/adam-A-bitwise-$X.json"
  check "adam-B-floor-$X" python -u compare_phase.py floor "$WORK/adam/A-clean-$X" "$WORK/adam/B-clean-$X" "$CSV" "$R/adam-B-floor-$X.json"
done

# 4. Polish: arm A bitwise; every polish against the thesis final model
for X in $POLISH_RUNS; do
  check "polish-A-bitwise-$X" python -u compare_phase.py bitwise "$WORK/polish/A-mainline-$X" "$WORK/polish/A-clean-$X" "$R/polish-A-bitwise-$X.json"
  for V in A-mainline A-clean B-clean; do
    check "polish-thesis-$V-$X" python -u compare_phase.py thesis "$WORK/polish/$V-$X" "$WORK/staging" "$R/polish-thesis-$V-$X.json"
  done
done

# 5. Compiled GPU smokes (PS2 opening and the compiled path), arm A
[ "$GPU_SMOKES" = 1 ] && check gpu-A-smokes python -u compare_gpu.py "$WORK/gpu"

# 6. Coverage of the arm-A clean tasks over the grey-box files (lines no phase ran; informational)
if [ "$COVERAGE" = 1 ]; then
  python -m coverage combine --data-file="$WORK/cov/.coverage" "$WORK"/cov/.coverage.* > "$R/coverage-combine.log" 2>&1
  python -m coverage report -m --data-file="$WORK/cov/.coverage" \
    --include="$COPY-reverted/gantry_interconnect_dynamic.py,$COPY-reverted/gantry_dynamic/*,$COPY-reverted/model_augmentation/*" \
    > "$R/coverage.txt" 2>&1
  row coverage INFO "$(tail -1 "$R/coverage.txt") (missed lines per file: reports/coverage.txt)"
fi

N_FAIL=$(grep -c ' FAIL ' "$SUM")
echo "PHASE_PROOF $([ "$N_FAIL" = 0 ] && echo PASS || echo "FAIL ($N_FAIL rows)")" >> "$SUM"
cat "$SUM"
