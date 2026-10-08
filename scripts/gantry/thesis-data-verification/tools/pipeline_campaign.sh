#!/bin/bash
# pipeline_campaign.sh: the full campaign, unattended, in stages (one block of the manifest per stage),
# one MATLAB job at a time, each resumable and retried after an emergency RAM kill (run_matlab.sh).
# GATE (REVIEW.md finding 1): starts only if logs/pipeline_pilot.log holds a PASS line for every check
# run on the final code (C2, C4a, C6, C7, C9, C10a, C10b, C11, C13 and C3/C4b/C8/C14 on the pilot);
# otherwise it logs what is missing and stops without generating anything.
# Before each stage the free disk is logged; the driver itself stops a stage below 2 GB free.
# After the stages: check_records on every saved record, then the noise figure.
# Log: logs/pipeline_campaign.log. MATLAB flags from logs/matlab_flags.txt (written by the pilot pipeline).
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
LOG=logs/pipeline_campaign.log
export RETRIES="${RETRIES:-60}" RETRY_WAIT="${RETRY_WAIT:-120}"
FLAGS=$(cat logs/matlab_flags.txt 2>/dev/null || echo "")
say() { echo "[campaign $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() { ./tools/run_matlab.sh "$@" 2>&1 | grep --line-buffered -v "^\[wd\] t=" | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "start; MATLAB flags '$FLAGS'"
missing=""
for pat in "C2  noise calibration, .* -> PASS" "C4a constructed multisine .* -> PASS" "C6  controller gate .* -> PASS" \
           "C7  .* -> PASS" "C9  every manifest reference .* -> PASS" "C10a training references.* -> PASS" \
           "C10b route class .* -> PASS" "C11 truth linearisation .* -> PASS" "C13 noise synthesis covariance .* -> PASS" \
           "C3 / C4b / C8 / C14 on .* saved records -> PASS"; do
  grep -q -e "$pat" logs/pipeline_pilot.log 2>/dev/null || missing="$missing | $pat"
done
if [ -n "$missing" ]; then
  say "GATE CLOSED, no generation. Missing PASS lines:$missing"
  exit 4
fi
say "gate open: every check on the final code passed"
# Stage order: the blocks of the manifest (one realisation of training and validation and no
# noise-only twins, user decisions 2026-09-27).
for stage in training validation test-I test-E frf; do
  say "stage $stage, C: free $(df -h /c | tail -1 | awk '{print $4}')"
  MATLAB_FLAGS=$FLAGS run "gen_$stage" "generate_thesis_data('$stage')"
done
say "all stages run; checking every saved record"
MATLAB_FLAGS=$FLAGS run check_records_all "addpath('checks'); check_records"
conda run -n GraduationProject python checks/plot_noise_calibration.py 2>&1 | tee -a "$LOG"
say "done, C: free $(df -h /c | tail -1 | awk '{print $4}')"
