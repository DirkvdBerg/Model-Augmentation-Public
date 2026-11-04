#!/bin/bash
# D-295 final batch: what the first run (W_OLD) left open, against its mainline outputs. Run on the login node from the
# harness folder, env active:  bash server_final.sh
# Trees (copies beside Clean/, so their data paths resolve as Clean/'s): COPY-final = Clean/ as it is now; COPY-pinv =
# tree B, Clean/ with only the P^-T of D-294 reverted (pinv_revert.py), for every check that is bitwise against the
# thesis code through the normalisation or the OBC tuples. Submits the CPU, large-memory and GPU arrays
# (server_final.sbatch) and the summary (server_final_summary.sbatch). Clean/ may be edited while the jobs run.

set -euo pipefail

# ---- settings
REPO=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
W_OLD=$REPO/simulations/clean-golden/phase-D295/20261103-2347        # first run: staging and mainline outputs
LABEL=$(date +%Y%m%d-%H%M)
WORK=$REPO/simulations/clean-golden/phase-D295/$LABEL-final
COPY=$REPO/Clean-D295-$LABEL
CSV=$REPO/Thesis-writeup/Results/24-10-2026/CSV
DUMPS=$HOME/.deepSI/checkpoints
BB_RUN=$REPO/scripts/gantry/blackbox-cl-trainable/outputs/search/lowpass/thesis_r8_grid/9_nx16_w16_lr1e-3_n80000_89756
TIME=08:00:00                                                       # every job ends before the next maintenance
# ----

cd "$REPO/Thesis-writeup/Code/Verification/clean-code"
for P in "$WORK" "$COPY-final" "$COPY-pinv"; do [ -e "$P" ] && { echo "$P exists"; exit 1; }; done
[ -d "$W_OLD/staging" ] || { echo "no staging in $W_OLD"; exit 1; }
mkdir -p "$WORK/logs" "$WORK/reports" "$WORK/gpu"

# 1. The two trees and their manifests
for T in final pinv; do rsync -a --exclude __pycache__ "$REPO/Clean/" "$COPY-$T/"; done
HARNESS_CLEAN_ROOT="$COPY-pinv" python -u pinv_revert.py apply
for T in final pinv; do (cd "$COPY-$T" && find . -name '*.py' | sort | xargs sha256sum) > "$WORK/clean-$T.sha256"; done
printf 'COPY=%q\nW_OLD=%q\nCSV=%q\n' "$COPY" "$W_OLD" "$CSV" > "$WORK/settings.sh"

# 2. Task lists: name, tree, command (run with bash in the harness folder)
PIN='export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2;'
GPUENV="export HARNESS_THREADS=4 TORCHINDUCTOR_CACHE_DIR=/dataB1/dirk_van_den_berg/torchinductor-cache/final TRITON_CACHE_DIR=/dataB1/dirk_van_den_berg/triton-cache/final;"
L3ENV="export HARNESS_GOLDEN=$WORK/l3 HARNESS_CLEAN_OUT=$WORK/l3/_clean;"
: > "$WORK/tasks-cpu.tsv"; : > "$WORK/tasks-big.tsv"; : > "$WORK/tasks-gpu.tsv"
add() { printf '%s\t%s\t%s\n' "$2" "$3" "$4" >> "$WORK/tasks-$1.tsv"; }
for R in R1 R4 R13; do
  add cpu "l4b-P-$R" pinv "$PIN python -u l4_loader.py --side clean --staging $W_OLD/staging --out $WORK/l4P/$R --runs $R --parts l4b"
done
add cpu adam0-P-R1 pinv "$PIN python -u phase_run.py --side clean --phase adam --run R1 --staging $W_OLD/staging --dumps $DUMPS --out $WORK/adam0/P-clean-R1 --n-its 50 --zero-moments"
for ROW in 221 226; do
  add cpu "l3-mainline-$ROW" none "$PIN $L3ENV python -u run_l3.py --side mainline --row $ROW --noise lowpass"
  add cpu "l3-P-$ROW" pinv "$PIN $L3ENV python -u run_l3.py --side clean --row $ROW --noise lowpass"
done
add cpu eval-R1 final "$PIN python -u eval_check.py $W_OLD/staging/89792 $WORK/eval/R1 $CSV R1"
add cpu eval-R6 final "$PIN python -u eval_check.py $W_OLD/staging/89799 $WORK/eval/R6 $CSV R6"
add cpu bb final "$PIN export THESIS_BB_DIR=$BB_RUN; python -u bb_mainline.py $WORK/bb-main && python -u bb_check.py $BB_RUN $WORK/bb-clean"
add big polish-P-R13 pinv "$PIN python -u phase_run.py --side clean --phase polish --run R13 --staging $W_OLD/staging --out $WORK/polish/P-clean-R13"
# compare_gpu.py also reads each side's log as gpu/logs/a-<row>-<side>.log
GL=$WORK/gpu/logs
add gpu gpu-P-221 pinv "set -o pipefail; $GPUENV python -u gpu_smoke.py --side clean --row 221 --out $WORK/gpu/a-221-clean 2>&1 | tee $GL/a-221-clean.log"
add gpu gpu-P-226 pinv "set -o pipefail; $GPUENV python -u gpu_smoke.py --side clean --row 226 --out $WORK/gpu/a-226-clean 2>&1 | tee $GL/a-226-clean.log && python -u gpu_smoke.py --side clean --row 226 --polish-from $WORK/gpu/a-226-clean/noisy/gpuA-226-lowpass --out $WORK/gpu/b-226-clean 2>&1 | tee $GL/b-226-clean.log"
# the GPU comparison reads both sides from one folder: mainline's smokes and logs of the first run
mkdir -p "$GL"
for C in a-221 a-226 b-226; do
  ln -s "$W_OLD/gpu/$C-mainline" "$WORK/gpu/$C-mainline"
  ln -s "$W_OLD/gpu/logs/$C-mainline.log" "$GL/$C-mainline.log"
done

# 3. Submit
n() { wc -l < "$WORK/tasks-$1.tsv"; }
SB="sbatch --parsable -t $TIME --export=ALL,WORK=$WORK"
A=$($SB,TASKS=tasks-cpu.tsv --array=0-$(( $(n cpu) - 1 )) -p kauai,molokai,lanai,oahu,hawaii -c 1 --mem=32gb server_final.sbatch)
B=$($SB,TASKS=tasks-big.tsv --array=0-$(( $(n big) - 1 )) -p kauai,molokai,lanai,oahu,hawaii -c 1 --mem=128gb server_final.sbatch)
G=$($SB,TASKS=tasks-gpu.tsv --array=0-$(( $(n gpu) - 1 )) -p hawaii,oahu -c 4 --gres=gpu:1 --mem=64gb server_final.sbatch)
S=$(sbatch --parsable -t 02:00:00 --dependency=afterany:$A:$B:$G --export=ALL,WORK="$WORK" server_final_summary.sbatch)
echo "work=$WORK cpu=$A big=$B gpu=$G summary=$S"
echo "When $S has finished: cat $WORK/summary.txt"
