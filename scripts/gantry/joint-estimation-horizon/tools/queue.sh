#!/usr/bin/env bash
# Run several runs ONE AT A TIME, each through wait_run.sh (RAM gate) and run.sh (watchdog), and
# stop at the first non-zero exit. Each argument is "<run_name>|<script> [args]".
#   bash scripts/gantry/joint-estimation-horizon/tools/queue.sh "g1_checks|info/g1_info.py --checks-only" ...
set -u
JH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for item in "$@"; do
  name="${item%%|*}"; rest="${item#*|}"
  echo "[queue] >>> ${name}: ${rest}"
  # shellcheck disable=SC2086
  bash "$JH/tools/wait_run.sh" "$name" $rest
  rc=$?
  echo "[queue] <<< ${name} exit ${rc}"
  [ "$rc" -ne 0 ] && { echo "[queue] stopping at ${name}"; exit "$rc"; }
done
echo "[queue] all done"
