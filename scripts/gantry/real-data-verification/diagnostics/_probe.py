"""Probe import chain step by step to find where run_telica_param_recovery.py hangs."""
import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.dirname(__file__))

import builtins
_print = builtins.print
def print(*a, **kw):
    _print(*a, **kw, flush=True)

print("STEP 1: sys.path set")

import matplotlib
matplotlib.use('Agg')
print("STEP 2: matplotlib OK")

import torch
print("STEP 3: torch OK")

from telica_loader import load_telica_log
print("STEP 4: telica_loader OK")

import lpv_lfr_baseline.scripts.precompute as _precompute
print("STEP 5: precompute OK")

import lpv_lfr_baseline.scripts.train_param_recovery as tr
print(f"STEP 6: train_param_recovery OK (EPOCHS={tr.EPOCHS})")

from lpv_lfr_baseline.scripts.precompute import _build_state_traj_logical
print("STEP 7: _build_state_traj_logical OK")

from lpv_lfr_baseline.scripts.train_param_recovery import _run_no_grad
print("STEP 8: _run_no_grad OK")

from lpv_lfr_baseline.core.physics import P as _P, ts as _ts
print(f"STEP 9: physics OK (ts={_ts})")

print("ALL IMPORTS OK")
