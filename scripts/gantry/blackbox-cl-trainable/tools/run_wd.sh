#!/usr/bin/env bash
# Launch one job under the watchdog (BB2-009/BB2-010): env python called directly (no conda run
# wrapper), waits until available RAM >= $NEED_MB (poll 30 s, give up after $MAX_WAIT_S).
# Usage: tools/run_wd.sh <out_dir_rel> <python args...>
set -u
OUTREL=$1; shift
NEED_MB=${NEED_MB:-2500}; MAX_WAIT_S=${MAX_WAIT_S:-10800}
ENV='C:\Users\20203253\AppData\Local\anaconda3\envs\GraduationProject'
DIRW='C:\Users\20203253\OneDrive - TU Eindhoven\Graduation Project\Baseline FP model\Baseline-LPV-Augmentation\scripts\gantry\blackbox-cl-trainable'
waited=0
while :; do
  av=$(powershell.exe -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue" | tr -d '\r')
  if [ "$av" -ge "$NEED_MB" ]; then echo "[run_wd] available ${av} MB >= ${NEED_MB} after ${waited} s"; break; fi
  if [ "$waited" -ge "$MAX_WAIT_S" ]; then echo "[run_wd] GAVE UP: available ${av} MB < ${NEED_MB} MB for ${waited} s"; exit 3; fi
  if [ $((waited % 600)) -eq 0 ]; then echo "[run_wd] waiting: available ${av} MB < ${NEED_MB} MB (${waited} s)"; fi
  sleep 30; waited=$((waited + 30))
done
RUN="cd /d \"$DIRW\" && set PYTHONDONTWRITEBYTECODE=1&& set PATH=$ENV;$ENV\Library\mingw-w64\bin;$ENV\Library\usr\bin;$ENV\Library\bin;$ENV\Scripts;%PATH%&& \"$ENV\python.exe\" -u $*"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/watchdog.ps1 -Run "$RUN" -Out "$OUTREL" \
  -RamKillGB 1.0 -RamAlertGB 1.5 -DiskKillGB 0.5 -DiskAlertGB 1.0
