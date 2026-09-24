#!/usr/bin/env bash
# Wait until available RAM >= WAIT_RAM_GB (default 3.6) for two consecutive 30 s polls, then hand
# over to tools/run.sh with the same arguments (which re-checks and applies the watchdog, JH-001).
#   bash scripts/gantry/joint-estimation-horizon/tools/wait_run.sh <run_name> <script> [args...]
# Gives up after WAIT_MAX_S (default 6 h) with exit 4.
set -u
JH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TH="${WAIT_RAM_GB:-3.6}"
MAX="${WAIT_MAX_S:-21600}"
ok=0; t=0
while :; do
  RAM=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)" | tr -d '\r')
  if awk "BEGIN{exit !(${RAM} >= ${TH})}"; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge 2 ] && break
  [ "$t" -ge "$MAX" ] && { echo "[wait_run] gave up after ${t} s, RAM ${RAM} GB"; exit 4; }
  sleep 30; t=$((t+30))
done
echo "[wait_run] RAM ${RAM} GB >= ${TH} GB after ${t} s; launching $1"
exec bash "$JH/tools/run.sh" "$@"
