#!/usr/bin/env bash
# Launch a blackbox-closed-loop python script under the watchdog, from any cwd (BB-001).
#   bash scripts/gantry/blackbox-closed-loop/tools/run.sh <run_name> <script relative to blackbox-closed-loop> [args...]
# Refuses to launch below MIN_RAM_GB available RAM (handoff section 13: 4 GB). Output folder:
# scripts/gantry/blackbox-closed-loop/outputs/<run_name> (run.log, resources.csv, watchdog_summary.txt).
# Adapted from scripts/gantry/encoder-transient/tools/run.sh (itself from telica-real/tools/run.sh).
set -u
BB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$BB/../../.." && pwd)"
RUN="$1"; shift
SCRIPT="$1"; shift
MIN_RAM_GB="${MIN_RAM_GB:-4.0}"
cd "$REPO" || exit 2
RAM=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)" | tr -d '\r')
FREE=$(powershell -NoProfile -Command "[math]::Round((Get-PSDrive C).Free/1GB,2)" | tr -d '\r')
echo "[run.sh] ${RUN}: available RAM ${RAM} GB, C: free ${FREE} GB before launch"
if awk "BEGIN{exit !(${RAM} < ${MIN_RAM_GB})}"; then
  echo "[run.sh] REFUSED: available RAM ${RAM} GB < ${MIN_RAM_GB} GB (handoff section 13)"; exit 3
fi
mkdir -p "$BB/outputs/${RUN}"
# PYTHONDONTWRITEBYTECODE: importing repo modules must not write __pycache__ outside this folder (BB-001).
export PYTHONDONTWRITEBYTECODE=1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gantry/blackbox-closed-loop/tools/watchdog.ps1 \
  -Run "set PYTHONDONTWRITEBYTECODE=1&& conda run --no-capture-output -n GraduationProject python -u scripts/gantry/blackbox-closed-loop/${SCRIPT} $*" \
  -Out "scripts/gantry/blackbox-closed-loop/outputs/${RUN}" ${WD_FLAGS:-} 2>&1 \
  | grep --line-buffered -v -E "^Gym has been|gymnasium|Please upgrade to Gym|Users of this version of Gym|See the migration guide" \
  | tee "$BB/outputs/${RUN}/run.log"
exit "${PIPESTATUS[0]}"
