#!/bin/bash
# run_matlab.sh <name> <matlab statement>
# One MATLAB batch job from this folder under the shared watchdog
# (scripts/gantry/thesis-data-verification/tools/watchdog.ps1: kills the tree below 1.0 GB free RAM).
# Output streams to stdout and outputs/logs/<name>/run.log. Refuses to start beside another MATLAB
# (one job at a time). The statement is followed by a completion marker assembled from two parts;
# R2025a batch can finish and never exit, so the tree is ended 20 s after the marker (same rule as
# thesis-data-verification/tools/run_matlab.sh, whose logic this copies without the retry loop).
# MATLAB_FLAGS (env) is passed to matlab, default -nojvm (no Simulink, no figures needed here).
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WD="$(cd "$HERE/../thesis-data-verification/tools" && pwd)/watchdog.ps1"
NAME="$1"; shift
STMT="$*"
MARK='@@TDG_DONE@@'
if tasklist //FI "IMAGENAME eq MATLAB.exe" 2>/dev/null | grep -qi "MATLAB.exe"; then
  echo "[run] another MATLAB is running: not starting $NAME"; exit 3
fi
mkdir -p "$HERE/outputs/logs/$NAME"
cd "$HERE"
LOG="outputs/logs/$NAME/run.log"
: > "$LOG"
powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$WD")" \
  -Run "matlab ${MATLAB_FLAGS:--nojvm} -batch \"cd('$(cygpath -w "$HERE")'); $STMT; fprintf('%s%s\\n', '@@TDG_', 'DONE@@');\"" \
  -Out "outputs/logs/$NAME" > "$LOG" 2>&1 &
WPID=$!
tail -n +1 -F "$LOG" 2>/dev/null &
TPID=$!
forced=0
while kill -0 "$WPID" 2>/dev/null; do
  if grep -q "$MARK" "$LOG" 2>/dev/null; then
    for i in $(seq 1 20); do kill -0 "$WPID" 2>/dev/null || break; sleep 1; done
    if kill -0 "$WPID" 2>/dev/null; then
      WIN=$(cat /proc/$WPID/winpid 2>/dev/null)
      echo "[run] $NAME: statement completed but MATLAB did not exit within 20 s; ending its tree (winpid $WIN)"
      [ -n "$WIN" ] && taskkill //T //F //PID "$WIN" > /dev/null 2>&1
      forced=1
    fi
    break
  fi
  sleep 3
done
wait "$WPID"; rc=$?
sleep 1; kill "$TPID" 2>/dev/null
if [ "$forced" -eq 1 ] || grep -q "$MARK" "$LOG"; then rc=0; fi
echo "[run] $NAME finished with code $rc"
exit $rc
