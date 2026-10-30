#!/bin/bash
# The full clean-side suite (D-284 step 4 item 8), one numerical job at a time, against the mainline goldens:
# layers 1 and 2 (16 cases), layer 3 (5), 3b (2), 3c (6 with goldens, 3 clean-only; after layer 3, whose clean
# outputs they resume from), each layer's comparison, and the CLI smoke. Outputs go to simulations/clean-golden/_clean.
# usage: bash run_clean_suite.sh
cd "$(dirname "$0")"
bash run_l12_all.sh clean
bash run_seq.sh \
  "compare.py l12" \
  "run_l3.py --side clean --row 221 --noise lowpass" \
  "run_l3.py --side clean --row 229 --noise noisefree" \
  "run_l3.py --side clean --row 224 --noise noisefree" \
  "run_l3.py --side clean --row 226 --noise lowpass" \
  "run_l3.py --side clean --row 227 --noise lowpass" \
  "compare.py l3" \
  "run_l3bc.py --side clean --layer 3b --row 221 --noise lowpass" \
  "run_l3bc.py --side clean --layer 3b --row 226 --noise lowpass" \
  "compare.py l3b" \
  "run_l3bc.py --side clean --layer 3c --row 226 --noise lowpass --kind i" \
  "run_l3bc.py --side clean --layer 3c --row 226 --noise lowpass --kind ii" \
  "run_l3bc.py --side clean --layer 3c --row 226 --noise lowpass --kind iii" \
  "run_l3bc.py --side clean --layer 3c --row 226 --noise lowpass --kind iv" \
  "run_l3bc.py --side clean --layer 3c --row 221 --noise lowpass --kind i" \
  "run_l3bc.py --side clean --layer 3c --row 221 --noise lowpass --kind ii" \
  "run_l3bc.py --side clean --layer 3c --row 226 --noise lowpass --kind v" \
  "run_l3bc.py --side clean --layer 3c --row 224 --noise noisefree --kind vi" \
  "run_l3bc.py --side clean --layer 3c --row 226 --noise lowpass --kind vii" \
  "compare.py l3c" \
  "../../../../Clean/gantry_interconnect_dynamic.py R6 lowpass --cpu --smoke"
echo "CLEAN_SUITE_DONE"
