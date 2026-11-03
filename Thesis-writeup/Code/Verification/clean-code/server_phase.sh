#!/bin/bash
# D-295: the server proof of Clean/ per phase, from the thesis checkpoints. Run on the login node, env active:
#   bash server_phase.sh
# It copies Clean/ twice beside it in the repository root (COPY-final for arm B; COPY-reverted, with the D-294 lines
# reverted, for arm A; beside Clean/ so that their data and output folders resolve as Clean/'s do), stages the thesis
# runs, writes the task list and submits: the CPU array (server_phase.sbatch), the GPU smokes
# (server_gpu_smoke.sbatch) and the summary (server_phase_summary.sbatch). Every job imports one of the two copies,
# never the repository's Clean/, so Clean/ may be edited while the jobs run; the proof is of the copies, whose
# manifests are in WORK. Delete the two copies once the summary is read.

set -euo pipefail

# ---- settings
REPO=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
LABEL=$(date +%Y%m%d-%H%M)
WORK=$REPO/simulations/clean-golden/phase-D295/$LABEL              # every output of this proof
COPY=$REPO/Clean-D295-$LABEL                                        # the two tree copies: $COPY-final, $COPY-reverted
THESIS_RUNS=$REPO/simulations/gantry_subnet/thesis_taf_lowpass_noise_data_lsq   # the nine final run folders
DUMPS=$HOME/.deepSI/checkpoints                                     # the thesis deepSI dumps (adam phase)
BB_RUN=$REPO/scripts/gantry/blackbox-cl-trainable/outputs/search/lowpass/thesis_r8_grid/9_nx16_w16_lr1e-3_n80000_89756
CSV=$REPO/Thesis-writeup/Results/24-10-2026/CSV                     # the report's numbers
L4_RUNS="R1 R2 R3 R4 R5 R6 R7 R13 R14"                              # evaluation: every final grey-box run
ADAM_RUNS="R1 R4 R6"                     # Adam resume: OBC + PS2, U detuned + PS2, U with n_a = 0
POLISH_RUNS="R3 R4 R13"                  # full polish: kept (U), rolled back (U), kept (OBC)
N_ITS=50                                 # Adam updates after the resume
COVERAGE=1                               # 1: the arm-A clean tasks run under coverage
GPU_SMOKES=1                             # 1: also the compiled GPU smokes (PS2 opening, compiled path; about 1 h)
# ----

H=$REPO/Thesis-writeup/Code/Verification/clean-code
cd "$H"
[ -d "$CSV" ] || { echo "no report CSVs at $CSV"; exit 1; }
[ -d "$DUMPS" ] || { echo "no deepSI dumps at $DUMPS"; exit 1; }
for P in "$WORK" "$COPY-final" "$COPY-reverted"; do [ -e "$P" ] && { echo "$P exists"; exit 1; }; done
python -u d294_revert.py status --expect final
mkdir -p "$WORK/logs" "$WORK/reports" "$WORK/cov"
(cd "$REPO" && git rev-parse HEAD && git status --short Clean) > "$WORK/git.txt"

# 1. The two trees: arm B the final tree as it is, arm A the same with the D-294 lines reverted; each with a manifest
for ARM in final reverted; do
  rsync -a --exclude __pycache__ "$REPO/Clean/" "$COPY-$ARM/"
done
HARNESS_CLEAN_ROOT="$COPY-reverted" python -u d294_revert.py apply
HARNESS_CLEAN_ROOT="$COPY-reverted" python -u d294_revert.py status --expect reverted
for ARM in final reverted; do
  (cd "$COPY-$ARM" && find . -name '*.py' | sort | xargs sha256sum) > "$WORK/clean-$ARM.sha256"
done

# 2. Staging of the nine final runs (copies only; the server folders stay untouched)
python -u l4_stage.py --src "$THESIS_RUNS" --dst "$WORK/staging"

# 4. Task list: name, arm, coverage (0/1), script and arguments (run by python in the harness folder)
T=$WORK/tasks.tsv
: > "$T"
add() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$T"; }
for R in $L4_RUNS; do
  add "l4-A-mainline-$R" A 0 "l4_loader.py --side mainline --staging $WORK/staging --out $WORK/l4A/mainline/$R --runs $R --eval-check"
  add "l4-A-clean-$R" A $COVERAGE "l4_loader.py --side clean --staging $WORK/staging --out $WORK/l4A/clean/$R --runs $R"
  add "l4-B-clean-$R" B 0 "l4_loader.py --side clean --staging $WORK/staging --out $WORK/l4B/clean/$R --runs $R"
done
for R in $ADAM_RUNS; do
  add "adam-A-mainline-$R" A 0 "phase_run.py --side mainline --phase adam --run $R --staging $WORK/staging --dumps $DUMPS --out $WORK/adam/A-mainline-$R --n-its $N_ITS"
  add "adam-A-clean-$R" A $COVERAGE "phase_run.py --side clean --phase adam --run $R --staging $WORK/staging --dumps $DUMPS --out $WORK/adam/A-clean-$R --n-its $N_ITS"
  add "adam-B-clean-$R" B 0 "phase_run.py --side clean --phase adam --run $R --staging $WORK/staging --dumps $DUMPS --out $WORK/adam/B-clean-$R --n-its $N_ITS"
done
for R in $POLISH_RUNS; do
  add "polish-A-mainline-$R" A 0 "phase_run.py --side mainline --phase polish --run $R --staging $WORK/staging --out $WORK/polish/A-mainline-$R"
  add "polish-A-clean-$R" A $COVERAGE "phase_run.py --side clean --phase polish --run $R --staging $WORK/staging --out $WORK/polish/A-clean-$R"
  add "polish-B-clean-$R" B 0 "phase_run.py --side clean --phase polish --run $R --staging $WORK/staging --out $WORK/polish/B-clean-$R"
done
add "bb-B" B 0 "-m blackbox.eval_bb $BB_RUN $WORK/bb"         # a module of the copy (PYTHONPATH, arm B)
N=$(wc -l < "$T")
printf 'L4_RUNS=%q\nADAM_RUNS=%q\nPOLISH_RUNS=%q\nCSV=%q\nGPU_SMOKES=%s\nCOVERAGE=%s\nCOPY=%q\n' \
  "$L4_RUNS" "$ADAM_RUNS" "$POLISH_RUNS" "$CSV" "$GPU_SMOKES" "$COVERAGE" "$COPY" > "$WORK/settings.sh"

# 5. Submit
A=$(sbatch --parsable --array=0-$((N - 1)) --export=ALL,WORK="$WORK" server_phase.sbatch)
DEP=$A
if [ "$GPU_SMOKES" = 1 ]; then
  G=$(sbatch --parsable --array=0-1 --export=ALL,OUT="$WORK/gpu",HARNESS_CLEAN_ROOT="$COPY-reverted" \
      server_gpu_smoke.sbatch)
  DEP=$A:$G
fi
S=$(sbatch --parsable --dependency=afterany:$DEP --export=ALL,WORK="$WORK" server_phase_summary.sbatch)
echo "work=$WORK tasks=$N cpu_array=$A gpu=${G:-none} summary=$S"
echo "When $S has finished: cat $WORK/summary.txt"
