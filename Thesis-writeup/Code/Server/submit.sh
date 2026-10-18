#!/bin/bash
# Submit one tier of the thesis campaign (Thesis-writeup/Documentation/TRAINING-DESIGN.md section 4)
# as a throttled job array on hawaii and oahu. Run from anywhere on the cluster:
#
# FINAL CAMPAIGN (THESIS-RESULTS.md section 4, final settings 2026-10-16; lowpass data, seed 1, mlp +
# live g_aug; the entry's THESIS block holds the final method, tasks/final-run-code-changes-2026-10-16.md).
# Logs: $LOG_DIR, default /home/dirk_van_den_berg/logs/augmentation/thesis-results-4. In this order:
#   bash .../submit.sh final-smoke     the gate: GPU launch test of every final row type (ids 231-240), one job
#   bash .../submit.sh final-all       everything, main results first: R1 (221) and R4 (224), the R8
#                                      black-box grid, then the appendix runs R2, R13 (222, 229) and
#                                      R3, R5, R6, R7, R14 (223, 225-227, 230) at lower priority (--nice)
# or tier by tier: final-obc-main, final-u-main, final-bb, final-obc [PAR], final-u [PAR].
#   THESIS_BB_DIR=<selected R8 output folder> bash .../submit.sh final-eval
#                                      after the runs: evaluation/run_eval.sbatch (CPU array, every model on
#                                      every split), then make_results.py -> Thesis-writeup/Results/*.csv
# The entry's THESIS block now runs the FINAL method for every THESIS_ARM row, so the rows below
# (201-217, 2026-10-11 campaign) no longer reproduce their run if resubmitted.
#
# 2026-10-11 CAMPAIGN (lowpass data, seed 1, mlp + zero g_aug; rows carry arch, ps2 and init):
#   sbatch Thesis-writeup/Code/Server/greybox/smoke_all.sbatch   launch test of every row type (ids 211-217)
#   bash .../submit.sh thesis-obc [PAR]          R1, R2 (ids 201, 202): OBC + PS2, 72 h
#   bash .../submit.sh thesis-u [PAR]            R3 to R7 (ids 203-207): U arm, 24 h
# Partitions: PARTITION=<comma list> picks any partitions for the thesis tiers, e.g.
#   PARTITION=hawaii,lanai,molokai,oahu bash .../submit.sh thesis-u 5
# Without it: thesis-obc on hawaii,lanai,molokai (oahu is slow for OBC, PS2-METHOD 7.5), thesis-u on
# hawaii,oahu. WALL=<hh:mm:ss> overrides the wall time the same way. PAR caps the runs at once.
# The examples below are historical tiers. Current defaults do not reproduce
# their original architecture/initialization; liveg/livegps2 ids 141-156 are superseded;
# compare-mlp rows 163/164 now need THESIS_INIT=live (D-240).
#   bash Thesis-writeup/Code/Server/submit.sh smoke         3 short launch tests on hawaii
#   bash Thesis-writeup/Code/Server/submit.sh core [PAR]    the 13-run core set, ids 41 to 53 in priority order:
#                                                              R4/R6 seeds 1-3 (41-46), R2 0/4/8 states (47-49),
#                                                              noise-free pair (50-51), correct start (52-53)
#   THESIS_NITS=<n> bash .../submit.sh 4 [PAR]                 tier 4: window sweep 0.2 and 0.4 s (2 runs),
#                                                              n = pilot's plateau updates x 1.5 (TRAINING-DESIGN 1.1)
#   Tier "karnopp" (ids 1-28) ran on the superseded Karnopp data (array 86449); never resubmit it.
#   Extensions (seeds 4-5, more noise-free twins) get NEW ids; never reuse or renumber an id.
#   bash .../submit.sh ps2smoke                 D-236: PS2 launch test, both arms (THESIS_SMOKE=1 THESIS_PS2=1, hawaii)
#   bash .../submit.sh ps2 [PAR]                D-236: PS2 opening, U/OBC x seeds 1-3 (ids 111-116), THESIS_PS2=1
#   (ids 121-126, tier ps2ctl: retired, never resubmit; references are the core runs 41-46, D-236 correction)
#   bash .../submit.sh liveg [PAR]              D-239: paper ResNet, plain, U/OBC x seeds 1-3 (ids 141-146)
#   bash .../submit.sh livegps2 [PAR]           D-239: the matched PS2 arm (ids 151-156)
#   bash .../submit.sh compare-resnet [PAR]    live ResNet plain/PS2, lowpass, U, seed 1
#   bash .../submit.sh compare-mlp [PAR]       live tanh MLP plain/PS2, same settings
#
# PAR = at most this many runs of the tier at once (default 2), so the tier never claims every GPU.
# Logs: $LOG_DIR/<tier>_<array>_<run id>.out (the sbatch -o in the job files is overridden here).
set -eo pipefail
cd "$(dirname "$0")/../../.."
TABLE=Thesis-writeup/Code/Server/tables/runs.tsv
SBATCH_FILE=Thesis-writeup/Code/Server/greybox/run_thesis.sbatch
LOG_DIR="${LOG_DIR:-/home/dirk_van_den_berg/logs/augmentation/thesis-results-4}"   # D-271 relaunch; the Adam-start campaign is in thesis-results-3
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/%x_%A_%a.out"

TIER="${1:?give a tier: final-smoke, final-all, or one row tier}"
PAR="${2:-2}"
# Optional overrides for the thesis tiers: any partition list and wall time.
PART_OBC="${PARTITION:-hawaii,lanai,molokai}"; WALL_OBC="${WALL:-72:00:00}"
PART_U="${PARTITION:-hawaii,oahu}";            WALL_U="${WALL:-24:00:00}"
case "$TIER" in final-*) WALL_U="${WALL:-48:00:00}" ;; esac   # pre-fit + 200 epochs on oahu's 2080 Ti
# Appendix tiers queue behind the main results (Dirk 2026-10-16): lower scheduling priority.
NICE=""; case "$TIER" in final-obc|final-u) NICE="--nice=1000" ;; esac

# Final-campaign tiers that are not a set of rows in the run table.
case "$TIER" in
  final-smoke)
    sbatch -J final-smoke -t 04:00:00 -o "${LOG_DIR}/%x_%j.out" --export=ALL,SMOKE_TIER=final-smoke \
      Thesis-writeup/Code/Server/greybox/smoke_all.sbatch
    exit 0 ;;
  final-bb)
    # R8 (D-264): the whole grid at 80000 updates with the grey box's objective, CPU only.
    sbatch -J final-bb -p "${PARTITION:-kauai,molokai,lanai}" -t 24:00:00 --array=0-10 \
      -o "${LOG_DIR}/bb_search_%A_%a.out" \
      --export="ALL,BB_VERSION=lowpass,BB_GRID=thesis_r8_grid.conf,BB_NITS=80000,BB_SHARED_EXTRA=--na 30 --objective nrmse" \
      Thesis-writeup/Code/Server/blackbox/run_bb_search.sh
    exit 0 ;;
  final-eval)
    # After the runs: every model on every split (D-269), then the result tables. THESIS_BB_DIR is the
    # selected R8 configuration's output folder (scripts/gantry/blackbox-cl-trainable/outputs/...).
    : "${THESIS_BB_DIR:?set THESIS_BB_DIR to the output folder of the selected R8 configuration}"
    JID=$(sbatch --parsable --array=0-10 -o "$LOG" --export=ALL,THESIS_BB_DIR=${THESIS_BB_DIR} \
      Thesis-writeup/Code/Server/evaluation/run_eval.sbatch)
    echo "evaluation array ${JID}"
    sbatch -J final-results -p kauai,molokai,lanai -c 2 --mem=8gb -t 01:00:00 --dependency=afterok:${JID} \
      -o "${LOG_DIR}/%x_%j.out" --export=ALL,THESIS_BB_DIR=${THESIS_BB_DIR} \
      --wrap ". \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject && cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation && THESIS_EVAL=final python -u Thesis-writeup/Code/Evaluation/make_results.py"
    exit 0 ;;
  final-all)
    # Main results first (R1, R4, R8), then the appendix runs; each tier as below.
    bash "$0" final-obc-main 1
    bash "$0" final-u-main 1
    # the grid may already have been submitted on its own (it needs no gate)
    if squeue -h -u "$USER" -n final-bb | grep -q .; then
      echo "final-bb already queued or running: not submitted again"
    else
      bash "$0" final-bb
    fi
    bash "$0" final-obc 2
    bash "$0" final-u 5
    exit 0 ;;
esac
case "$TIER" in
  compare-resnet|compare-mlp) TABLE=Thesis-writeup/Code/Server/tables/live_compare.tsv ;;
esac
IDS=$(awk -F'\t' -v t="$TIER" 'NR>1 && $2==t {printf "%s%s", sep, $1; sep=","}' "$TABLE")
if [ -z "$IDS" ]; then echo "no runs for tier ${TIER} in ${TABLE}"; exit 2; fi
echo "tier ${TIER}: run ids ${IDS}"

case "$TIER" in
  thesis-obc|final-obc-main|final-obc)
    # OBC runs: 13 to 20 h on the A100 nodes (lanai, molokai), about 30 to 35 h on hawaii; blade1
    # (oahu) did not finish one in 45 h (runs 41, 43), so it is not in the default list.
    echo "partitions ${PART_OBC}, wall ${WALL_OBC}, at most ${PAR} at once"
    sbatch -J "${TIER}" -o "$LOG" $NICE -p "${PART_OBC}" -t "${WALL_OBC}" --array="${IDS}%${PAR}" \
      --export=ALL,RUN_TABLE=${TABLE} "$SBATCH_FILE" ;;
  thesis-u|final-u-main|final-u)
    echo "partitions ${PART_U}, wall ${WALL_U}, at most ${PAR} at once"
    sbatch -J "${TIER}" -o "$LOG" $NICE -p "${PART_U}" -t "${WALL_U}" --array="${IDS}%${PAR}" \
      --export=ALL,RUN_TABLE=${TABLE} "$SBATCH_FILE" ;;
  thesis-smoke|final-smoke)
    echo "${TIER} rows are run in one job: sbatch --export=ALL,SMOKE_TIER=${TIER} Thesis-writeup/Code/Server/greybox/smoke_all.sbatch"; exit 2 ;;
  smoke)
    # Short launch test (about 5 to 10 min): real data cut to 2 s, real GPU, eager, OBC, float64,
    # 4 updates, a 2-iteration polish and the end-of-run evaluation. Compilation is not in it:
    # watch the first real run's log for "training rollout COMPILED" and its first updates.
    sbatch -J thesis-smoke -p hawaii -t 00:30:00 --array="${IDS}" \
      --export=ALL,RUN_TABLE=${TABLE},THESIS_SMOKE=1 "$SBATCH_FILE" ;;
  ps2smoke)
    sbatch -J thesis-ps2smoke -p hawaii -t 00:30:00 --array="${IDS}" --export=ALL,RUN_TABLE=${TABLE},THESIS_SMOKE=1,THESIS_PS2=1 "$SBATCH_FILE" ;;
  ps2)
    sbatch -J thesis-ps2 --array="${IDS}%${PAR}" --export=ALL,RUN_TABLE=${TABLE},THESIS_PS2=1 "$SBATCH_FILE" ;;
  liveg)
    sbatch -J thesis-liveg --array="${IDS}%${PAR}" --export=ALL,RUN_TABLE=${TABLE},THESIS_PS2=0 "$SBATCH_FILE" ;;
  livegps2)
    sbatch -J thesis-livegps2 --array="${IDS}%${PAR}" --export=ALL,RUN_TABLE=${TABLE},THESIS_PS2=1 "$SBATCH_FILE" ;;
  4)
    : "${THESIS_NITS:?tier 4 needs THESIS_NITS = the pilot plateau updates x 1.5 (TRAINING-DESIGN 1.1)}"
    sbatch -J thesis-t4 -t 48:00:00 --array="${IDS}%${PAR}" \
      --export=ALL,RUN_TABLE=${TABLE},THESIS_NITS=${THESIS_NITS} "$SBATCH_FILE" ;;
  *)
    sbatch -J "thesis-t${TIER}" --array="${IDS}%${PAR}" \
      --export=ALL,RUN_TABLE=${TABLE} "$SBATCH_FILE" ;;
esac
