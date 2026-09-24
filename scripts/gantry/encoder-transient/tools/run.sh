#!/usr/bin/env bash
# Launch an encoder-transient python script under the watchdog, from any cwd.
#   bash scripts/gantry/encoder-transient/tools/run.sh <run_name> <script relative to encoder-transient> [args...]
# Output folder: scripts/gantry/encoder-transient/outputs/<run_name> (resources.csv, watchdog_summary.txt).
# Extra watchdog flags via the WD_FLAGS env var. Copied from telica-real/tools/run.sh (ET-001).
set -u
ET="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ET/../../.." && pwd)"
RUN="$1"; shift
SCRIPT="$1"; shift
cd "$REPO" || exit 2
FREE=$(powershell -NoProfile -Command "[math]::Round((Get-PSDrive C).Free/1GB,2)")
echo "[run.sh] C: free ${FREE} GB before launch of ${RUN}"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gantry/encoder-transient/tools/watchdog.ps1 \
  -Run "conda run --no-capture-output -n GraduationProject python -u scripts/gantry/encoder-transient/${SCRIPT} $*" \
  -Out "scripts/gantry/encoder-transient/outputs/${RUN}" ${WD_FLAGS:-} 2>&1 \
  | grep --line-buffered -v -E "^Gym has been|gymnasium|Please upgrade to Gym|Users of this version of Gym|See the migration guide"
