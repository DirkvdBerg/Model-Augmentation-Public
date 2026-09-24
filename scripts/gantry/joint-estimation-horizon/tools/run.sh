#!/usr/bin/env bash
# Launch a joint-estimation-horizon python script under the watchdog (JH-001), from any cwd.
#   bash scripts/gantry/joint-estimation-horizon/tools/run.sh <run_name> <script relative to the folder> [args...]
# Output folder: outputs/<run_name> (run.log, resources.csv, watchdog_summary.txt).
# Refuses to launch below MIN_RAM_GB available RAM (JH-001: 3.5 GB). Extra env (e.g. JH_FILESET)
# is inherited by the child. Adapted from telica-real/tools/run.sh.
set -u
JH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$JH/../../.." && pwd)"
RUN="$1"; shift
SCRIPT="$1"; shift
MIN_RAM_GB="${MIN_RAM_GB:-3.5}"
cd "$REPO" || exit 2
RAM=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)" | tr -d '\r')
FREE=$(powershell -NoProfile -Command "[math]::Round((Get-PSDrive C).Free/1GB,2)" | tr -d '\r')
echo "[run.sh] ${RUN}: available RAM ${RAM} GB, C: free ${FREE} GB before launch"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/[run.sh] GPU process: /'
if awk "BEGIN{exit !(${RAM} < ${MIN_RAM_GB})}"; then
  echo "[run.sh] REFUSED: available RAM ${RAM} GB < ${MIN_RAM_GB} GB (JH-001)"; exit 3
fi
mkdir -p "$JH/outputs/${RUN}"
export PYTHONDONTWRITEBYTECODE=1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gantry/joint-estimation-horizon/tools/watchdog.ps1 \
  -Run "set PYTHONDONTWRITEBYTECODE=1&& conda run --no-capture-output -n GraduationProject python -u scripts/gantry/joint-estimation-horizon/${SCRIPT} $*" \
  -Out "scripts/gantry/joint-estimation-horizon/outputs/${RUN}" ${WD_FLAGS:-} 2>&1 \
  | grep --line-buffered -v -E "^Gym has been|gymnasium|Please upgrade to Gym|Users of this version of Gym|See the migration guide" \
  | tee "$JH/outputs/${RUN}/run.log"
exit "${PIPESTATUS[0]}"
