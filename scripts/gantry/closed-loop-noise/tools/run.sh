#!/usr/bin/env bash
# Launch a closed-loop-noise python script under the watchdog (CN-001), from any cwd.
#   bash scripts/gantry/closed-loop-noise/tools/run.sh <run_name> <script relative to closed-loop-noise> [args...]
# MATLAB: CN_MATLAB=<dir relative to closed-loop-noise> bash .../run.sh <run_name> <function name>
# Output folder: scripts/gantry/closed-loop-noise/outputs/<run_name>. Extra watchdog flags via WD_FLAGS.
set -u
CN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$CN/../../.." && pwd)"
REL="scripts/gantry/closed-loop-noise"
RUN="$1"; shift
SCRIPT="$1"; shift
cd "$REPO" || exit 2
FREE=$(powershell -NoProfile -Command "[math]::Round((Get-PSDrive C).Free/1GB,2)")
echo "[run.sh] C: free ${FREE} GB before launch of ${RUN}"
if [ -n "${CN_MATLAB:-}" ]; then
  JVM="-nojvm"; [ -n "${CN_MATLAB_JVM:-}" ] && JVM=""
  CMD="matlab ${JVM} -sd ${REL}/${CN_MATLAB} -batch ${SCRIPT}"
else
  CMD="conda run --no-capture-output -n GraduationProject python -u ${REL}/${SCRIPT} $*"
fi
powershell -NoProfile -ExecutionPolicy Bypass -File ${REL}/tools/watchdog.ps1 \
  -Run "${CMD}" -Out "${REL}/outputs/${RUN}" ${WD_FLAGS:-} 2>&1 \
  | grep --line-buffered -v -E "^Gym has been|gymnasium|Please upgrade to Gym|Users of this version of Gym|See the migration guide"
