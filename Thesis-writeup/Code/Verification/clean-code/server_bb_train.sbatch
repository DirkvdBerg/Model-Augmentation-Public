#!/bin/bash
#SBATCH -J clean-bb-train
#SBATCH -p kauai,molokai,lanai,oahu,hawaii
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH -t 03:00:00
#SBATCH -o /home/dirk_van_den_berg/logs/augmentation/clean-code/%x_%j.out

# D-295: black-box training of R8 (grid index 9) by mainline (train_bb_clmap.py, the thesis R8 flags) and by Clean/
# (blackbox.train_bb, its defaults are R8's), on the same node, one after the other, then compare_bb.py bitwise.
# Full records and N_ITS updates on both sides: Clean/'s train_bb has no --truncate-s, and with N_ITS below the
# shared validation interval (1000) both validate at the same updates. Submit: sbatch --export=ALL,OUT=<folder>
# server_bb_train.sbatch  (optional N_ITS, default 200)

set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
REPO=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
cd "$REPO"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4 MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2
: "${OUT:?submit with --export=ALL,OUT=<folder>}"
N_ITS=${N_ITS:-200}
[ -e "$OUT" ] && { echo "$OUT exists"; exit 1; }
mkdir -p "$OUT"
echo "node=$SLURM_JOB_NODELIST n_its=$N_ITS"; date

# deepSI keeps its checkpoints in ~/.deepSI on Linux; compare_bb.py reads them from <run>/localappdata/...
collect() {
  local run=$1 name
  name=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['checkpoint'])" "$run/result.json")
  mkdir -p "$run/localappdata/deepSI/checkpoints"
  for s in _best _last; do
    cp "$HOME/.deepSI/checkpoints/${name%_best.pth}$s.pth" "$run/localappdata/deepSI/checkpoints/"
  done
}

python -u scripts/gantry/blackbox-cl-trainable/train_bb_clmap.py --server --version lowpass --seed 1 --target e --in64 \
  --f-blockmean --batch 256 --stride 25 --n-its "$N_ITS" --its-per-val 1000 --epochs 1000 --na 30 --objective nrmse \
  --nx 16 --fh-width 16 --e-width 32 --lr 1e-3 --out "$OUT/mainline" 2>&1 | tee "$OUT/mainline.log"
collect "$OUT/mainline"
(cd Clean && python -u -m blackbox.train_bb --n-its "$N_ITS" --out "$OUT/clean") 2>&1 | tee "$OUT/clean.log"
collect "$OUT/clean"
date
python -u Thesis-writeup/Code/Verification/clean-code/compare_bb.py "$OUT/mainline" "$OUT/clean" 2>&1 \
  | grep -v "FutureWarning\|torch.load" | tee "$OUT/compare.txt"
