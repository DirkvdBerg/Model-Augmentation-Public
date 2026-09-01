"""End-to-end wiring check: does compilation survive a real fit() call? (D-169)

Everything about the compiled path so far is measured on BENCHMARKS that call
`closed_loop_rollout` directly. This runs the real `train_model` -> deepSI `fit()` path with
`n_its` capped, which is the only way to exercise:

  * compilation actually engaging in training (not just in a benchmark)
  * fit()'s INITIAL validation, which runs BEFORE training and on the CPU under the old code --
    the exact sequence that left the connection matrices on the CPU and made inductor emit
    "skipping cudagraphs due to cpu device", silently degrading reduce-overhead to plain inductor
  * the per-validation device flip, which job 80695 measured at 599.56 s / 795.69 s for the first
    update afterwards, against a 3.30 s eager baseline, and which `fit()` no longer performs when
    a simulator is attached
  * the recompile detector in ClosedLoopSimulator staying quiet
  * config.json recording DEVICE / COMPILE_MODE / USE_F64

NOT the full entry point: `main()`'s post-training block runs four baseline sims plus per-record
NRMS over 8 records, roughly an hour of 48,000-step free runs that test nothing about the wiring.

Run:  python scripts/gantry/GPU/wiring_e2e.py
Env:  E2E_DEVICE (cuda), E2E_COMPILE (reduce-overhead; 'none'/'eager'/'off' = uncompiled),
      E2E_F64 (1),
      E2E_NF (400), E2E_BATCH (512), E2E_ITS (2)
"""
__project_origin__ = "added"

import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))

from gantry_dynamic.config import save_dir, config_json_dict, git_provenance     # noqa: E402
from gantry_dynamic.data import load_datasets, compute_normalization, VAL_FILES, TRAIN_FILES  # noqa: E402
from gantry_dynamic.model import build_model                                     # noqa: E402
from gantry_dynamic.controller import build_closed_loop                          # noqa: E402
from gantry_dynamic.training import train_model_with_diagnostics                 # noqa: E402
import gantry_interconnect_dynamic as entry                                      # noqa: E402

DEVICE  = os.environ.get('E2E_DEVICE', 'cuda')
_c = os.environ.get('E2E_COMPILE', 'reduce-overhead')
COMPILE = None if _c.lower() in ('', 'none', 'eager', 'off') else _c
USE_F64 = os.environ.get('E2E_F64', '1') not in ('0', 'false', 'False')
NF      = int(os.environ.get('E2E_NF', 400))
BATCH   = int(os.environ.get('E2E_BATCH', 512))
ITS     = int(os.environ.get('E2E_ITS', 2))


def banner(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


banner("SETUP")
if DEVICE == 'cuda' and not torch.cuda.is_available():
    print("device=cuda requested but no CUDA available (need sbatch --gres=gpu:1)")
    sys.exit(1)
print(f"  device={DEVICE}  compile_mode={COMPILE!r}  use_f64={USE_F64}  "
      f"nf={NF}  batch={BATCH}  n_its={ITS}")
if DEVICE == 'cuda':
    print(f"  GPU {torch.cuda.get_device_properties(0).name}")

torch.manual_seed(42)
np.random.seed(42)
cfg = replace(entry.CFG, orth=False, device=DEVICE, compile_mode=COMPILE, use_f64=USE_F64,
              checkpoint_chunk=0, nf_override=NF, batch_size=BATCH, n_its=ITS,
              stride=entry.CFG.stride, save_flag=False)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
fs = build_model(cfg.hp, cfg, data, norm)
fs.simulator = build_closed_loop(fs, norm, cfg, train_files=TRAIN_FILES, val_files=VAL_FILES,
                                 val_data=data.val_ckpt_data, verbose=True)

print("\n-- wiring, before training --")
print(f"  simulator._compiled is None : {fs.simulator._compiled is None}"
      f"   (False = the training rollout is compiled)")
print(f"  fit_sys.hfn is the RAW module: "
      f"{type(fs.hfn).__name__ != 'OptimizedModule'}   (True = validation stays eager)")
_cj = config_json_dict(cfg, git=git_provenance())
print(f"  config.json  DEVICE={_cj['DEVICE']}  COMPILE_MODE={_cj['COMPILE_MODE']}  "
      f"USE_F64={_cj['USE_F64']}  CHECKPOINT_CHUNK={_cj['CHECKPOINT_CHUNK']}")

banner(f"train_model  (n_its={ITS}; fit() validates once BEFORE training and once at the cap)")
# Through the same entry point the production run uses, so the wiring under test is the real one.
t0 = time.perf_counter()
bestfit, diag = train_model_with_diagnostics(fs, cfg.hp, cfg, data, norm,
                                             resume_ckpt=None, checkpoint_dir=None,
                                             run_id='e2e')
elapsed = time.perf_counter() - t0

banner("RESULT")
print(f"  completed in {elapsed:.1f} s   bestfit = {bestfit:.6e}")
hist = getattr(fs.simulator, '_t_hist', [])
if hist:
    print(f"  updates timed: {len(hist)}   "
          f"per-update s: {' '.join('%.2f' % v for v in hist)}")
    print(f"  max/median ratio: {max(hist) / sorted(hist)[len(hist) // 2]:.1f}x"
          f"   (a recompile shows as a large spike; the detector warns above 5x)")
print("\n  WHAT TO CHECK IN THE LOG ABOVE:")
print("   1. no 'skipping cudagraphs due to cpu device'  -> reduce-overhead kept its fast path")
print("   2. no '[compile] update took ... RECOMPILE'     -> the device flip is gone")
print("   3. '[closed loop] training rollout COMPILED'    -> compilation actually engaged")
print("\nwiring_e2e complete")
