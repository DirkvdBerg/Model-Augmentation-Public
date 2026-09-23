#!/usr/bin/env bash
# Launch a telica-real python script under the watchdog (TR-002), from any cwd.
#   bash telica-real/tools/run.sh <run_name> <script relative to telica-real> [args...]
# Output folder: telica-real/outputs/<run_name>. Extra watchdog flags via WD_FLAGS env var.
set -u
TR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$TR/.." && pwd)"
RUN="$1"; shift
SCRIPT="$1"; shift
cd "$REPO" || exit 2
FREE=$(powershell -NoProfile -Command "[math]::Round((Get-PSDrive C).Free/1GB,2)")
echo "[run.sh] C: free ${FREE} GB before launch of ${RUN}"
powershell -NoProfile -ExecutionPolicy Bypass -File telica-real/tools/watchdog.ps1 \
  -Run "conda run --no-capture-output -n GraduationProject python -u telica-real/${SCRIPT} $*" \
  -Out "telica-real/outputs/${RUN}" ${WD_FLAGS:-} 2>&1 \
  | grep --line-buffered -v -E "^Gym has been|gymnasium|Please upgrade to Gym|Users of this version of Gym|See the migration guide"
