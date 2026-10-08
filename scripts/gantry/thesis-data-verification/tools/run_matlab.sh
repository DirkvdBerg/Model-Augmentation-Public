#!/bin/bash
# run_matlab.sh <name> <matlab statement>
# Runs one MATLAB batch job from Code/Data under the watchdog; output streams to stdout and
# logs/<name>/attempt<k>.log; the watchdog's resource record is logs/<name>/resources.csv.
# One job at a time (handoff section 12): while any MATLAB runs, the job waits and tries again.
# MATLAB_FLAGS (env) is passed to matlab, e.g. MATLAB_FLAGS=-nojvm.
# RETRIES (env, default 0): if the watchdog kills the job (exit 137, emergency RAM floor 1.0 GB, set by
# the user), wait RETRY_WAIT seconds (default 120) and start it again, at most RETRIES times. This is
# not a launch gate on free RAM: every attempt starts at once; it only repeats a job the emergency
# kill stopped. Jobs must therefore be resumable (generate_thesis_data skips finished records and
# removes interrupted *_part.mat writes).
# COMPLETION MARKER: the statement is followed by fprintf('@@TDG_DONE@@'). MATLAB R2025a started with
# -nojvm was seen to hang at exit after finishing its work (2026-09-26, 0 % CPU for 10 min). When the
# marker has appeared and MATLAB has not exited 20 s later, this script ends the process tree and
# reports success: the marker proves the statement completed. A statement that errors exits MATLAB
# with code 1 and prints no marker, so a failure is never reported as success.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"          # scripts/gantry/thesis-data-verification
GEN="$(cd "$HERE/../../../Thesis-writeup/Code/Data" && pwd)"   # the generator, put on the MATLAB path
NAME="$1"; shift
STMT="$*"
RETRIES="${RETRIES:-0}"
RETRY_WAIT="${RETRY_WAIT:-120}"
MARK='@@TDG_DONE@@'
mkdir -p "$HERE/logs/$NAME"
cd "$HERE"
attempt=0
while true; do
  attempt=$((attempt + 1))
  if tasklist //FI "IMAGENAME eq MATLAB.exe" 2>/dev/null | grep -qi "MATLAB.exe"; then
    # never beside another MATLAB (this session's or another's): wait, then try again
    if [ "$attempt" -gt "$RETRIES" ]; then echo "[run] another MATLAB is running: giving up on $NAME"; exit 3; fi
    echo "[run] another MATLAB is running: $NAME waits $RETRY_WAIT s"
    sleep "$RETRY_WAIT"; continue
  fi
  echo "[run] attempt $attempt, C: free $(df -h /c | tail -1 | awk '{print $4}') before $NAME"
  ALOG="logs/$NAME/attempt$attempt.log"
  : > "$ALOG"
  powershell -NoProfile -ExecutionPolicy Bypass -File tools/watchdog.ps1 \
    -Run "matlab ${MATLAB_FLAGS:-} -batch \"cd('$(cygpath -w "$HERE")'); addpath('$(cygpath -w "$GEN")'); $STMT; fprintf('%s%s\\n', '@@TDG_', 'DONE@@');\"" -Out "logs/$NAME" > "$ALOG" 2>&1 &
  WPID=$!
  tail -n +1 -F "$ALOG" 2>/dev/null &
  TPID=$!
  forced=0
  while kill -0 "$WPID" 2>/dev/null; do
    if grep -q "$MARK" "$ALOG" 2>/dev/null; then
      for i in $(seq 1 20); do kill -0 "$WPID" 2>/dev/null || break; sleep 1; done
      if kill -0 "$WPID" 2>/dev/null; then
        WIN=$(cat /proc/$WPID/winpid 2>/dev/null)
        echo "[run] $NAME: statement completed but MATLAB did not exit within 20 s; ending its process tree (winpid $WIN)"
        [ -n "$WIN" ] && taskkill //T //F //PID "$WIN" > /dev/null 2>&1
        forced=1
      fi
      break
    fi
    sleep 3
  done
  wait "$WPID"; rc=$?
  sleep 1; kill "$TPID" 2>/dev/null
  # the marker is assembled in MATLAB from two parts, so the watchdog's echo of the command line
  # cannot contain it; once printed, the statement has completed, whatever ended the process
  if [ "$forced" -eq 1 ] || grep -q "$MARK" "$ALOG"; then rc=0; fi
  if [ "$rc" -ne 137 ] || [ "$attempt" -gt "$RETRIES" ]; then
    echo "[run] $NAME finished with code $rc after $attempt attempt(s)"
    exit $rc
  fi
  echo "[run] killed by the emergency RAM floor; retry in $RETRY_WAIT s"
  sleep "$RETRY_WAIT"
done
