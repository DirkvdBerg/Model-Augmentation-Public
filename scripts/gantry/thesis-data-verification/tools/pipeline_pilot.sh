#!/bin/bash
# pipeline_pilot.sh: the unattended pilot sequence, one MATLAB job at a time, each retried after an
# emergency RAM kill (run_matlab.sh). Log: logs/pipeline_pilot.log (every line, unbuffered).
#   1. C15 smoke (checks/nojvm_smoke.m) once without and once with the JVM; each run writes its own
#      checksum file logs/smoke_jvm0.txt / smoke_jvm1.txt; only if both exist and are identical do the
#      later jobs run with -nojvm (about 0.5 GB less RAM); otherwise with the JVM (REVIEW.md finding 2)
#   2. every check on the final code in ONE MATLAB job: C4a C7 C9 C10 (check_references), C6
#      (check_controller_gate), C2 C11 (check_noise_calibration), C13 (check_noise_draw)
#   3. the 14 pilot records in both versions, then C3 C4b C8 C14 (check_records) on them
# pipeline_campaign.sh starts the campaign only if every PASS line of this log is present.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
LOG=logs/pipeline_pilot.log
export RETRIES="${RETRIES:-60}" RETRY_WAIT="${RETRY_WAIT:-120}"
say() { echo "[pipeline $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() { ./tools/run_matlab.sh "$@" 2>&1 | grep --line-buffered -v "^\[wd\] t=" | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "start; RETRIES=$RETRIES RETRY_WAIT=$RETRY_WAIT"
rm -f logs/smoke_jvm0.txt logs/smoke_jvm1.txt
MATLAB_FLAGS=-nojvm run smoke_nojvm "addpath('checks'); nojvm_smoke"
MATLAB_FLAGS=      run smoke_jvm   "addpath('checks'); nojvm_smoke"
if [ -s logs/smoke_jvm0.txt ] && [ -s logs/smoke_jvm1.txt ] && cmp -s logs/smoke_jvm0.txt logs/smoke_jvm1.txt; then
  FLAGS=-nojvm; say "C15 smoke: -nojvm and JVM checksums identical ($(cat logs/smoke_jvm0.txt)) -> PASS, use -nojvm"
else
  FLAGS=; say "C15 smoke: files missing or different (jvm0 '$(cat logs/smoke_jvm0.txt 2>/dev/null)', jvm1 '$(cat logs/smoke_jvm1.txt 2>/dev/null)') -> keep the JVM"
fi
echo "$FLAGS" > logs/matlab_flags.txt

MATLAB_FLAGS=$FLAGS run checks_final "addpath('checks'); check_references; check_controller_gate; check_noise_calibration; check_noise_draw"
P="{'TR-P3_r1','TR-S1_r1','TR-Y1_r1','TR-L2_r1','TR-T1_r1','I3_r1','I4_r1','I5_r1','I6_r1','E1_r1','E5_r1','E6_r1','TF-3_r1'}"
MATLAB_FLAGS=$FLAGS run pilot "generate_thesis_data($P); addpath('checks'); check_records($P)"
say "done"
