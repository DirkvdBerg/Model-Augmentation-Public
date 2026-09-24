#!/usr/bin/env bash
# wd.sh <run-id> <command ...>
# Runs <command> under tools/watchdog.ps1 from the orthogonal-addition folder, live output to
# outputs/<run-id>.log (also echoed), watchdog CSV and summary in outputs/<run-id>/.
# Every script in this folder is launched through here (handoff sect. 9 G0, OA-001).
set -u
OA="$(cd "$(dirname "$0")/.." && pwd)"
cd "$OA" || exit 2
RID="$1"; shift
CMD="$*"
mkdir -p "outputs/$RID"
powershell -NoProfile -ExecutionPolicy Bypass -File tools/watchdog.ps1 \
    -Run "$CMD" -Out "outputs/$RID" 2>&1 | tee "outputs/$RID.log"
exit "${PIPESTATUS[0]}"
