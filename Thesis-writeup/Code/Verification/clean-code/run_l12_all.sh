#!/bin/bash
# Layers 1 and 2 for every case of design 6.3 on one side, one process per case, sequentially.
# usage: bash run_l12_all.sh mainline|clean [row-noise ...]
cd "$(dirname "$0")"
SIDE=$1; shift
CASES=${@:-"221-lowpass 222-lowpass 223-lowpass 224-lowpass 226-lowpass 227-lowpass 229-lowpass 230-lowpass 221-noisefree 222-noisefree 223-noisefree 224-noisefree 226-noisefree 227-noisefree 229-noisefree 230-noisefree"}
for c in $CASES; do
  row=${c%%-*}; noise=${c#*-}
  PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject \
    python -u capture_l12.py --side "$SIDE" --row "$row" --noise "$noise" 2>&1 | grep --line-buffered -av "FutureWarning\|torch.load(\|Gym\|gymnasium\|migration guide\|Please upgrade"
  echo "L12_CASE_END side=$SIDE case=$c rc=${PIPESTATUS[0]}"
done
echo "L12_ALL_DONE side=$SIDE"
