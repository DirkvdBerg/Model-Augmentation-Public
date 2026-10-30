#!/bin/bash
# Run harness jobs one after another (one numerical job at a time), output streamed line by line.
# usage: bash run_seq.sh "run_l3.py --side mainline --row 229 --noise noisefree" "compare.py l3 ..." ...
cd "$(dirname "$0")"
for job in "$@"; do
  echo "SEQ_START $job"
  T0=$(date +%s)
  PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject \
    python -u $job 2>&1 | grep --line-buffered -av "FutureWarning\|torch.load(\|Gym has been\|gymnasium\|migration guide\|Please upgrade\|Users of this version"
  echo "SEQ_END rc=${PIPESTATUS[0]} seconds=$(( $(date +%s) - T0 )) $job"
done
echo "SEQ_ALL_DONE"
