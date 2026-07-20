"""
make_drift_checkpoint.py -- train ONE X+Theta+Y config fast and save the drifted
_last checkpoint for d6 to dissect.

Why this exists: d6 needs the TRAINED (drifted) ANN, i.e. the _last checkpoint,
not _best (which reverts to epoch 0 = zero-init ANN). The Optuna search (job 69399)
never saved a usable one (killed before the final retrain; per-trial _best = epoch 0).
This reuses the search's fast setup -- cropped validation (8k, not the full 192k) and
stride=100 -- so it finishes in minutes, and it does NOT touch the experiment scripts.

Config = Trial 3 (lr=1.49e-8, nf=1400): the one search config whose val sim-RMS
actually moved past epoch 0 (6.58e-5 vs 8.01e-5), so its _last has a genuinely
trained ANN.

Checkpoint mechanics (framework quirk handled here):
  * fit() saves _best and _last, then RELOADS _best at the end (interconnect.py:714-716),
    so after train_model the in-memory model is _best (epoch 0). We reload _last to get
    the drifted model, then re-save it (whole __dict__, exactly as checkpoint_save_system
    does) to an explicit path d6 reads.

Per-epoch train/val nf-RMS is printed via the D-102 probe (mandatory for every gantry
training script).

Run:
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/make_drift_checkpoint.py
Env: EPOCHS (20), LR (1.49e-8), NF (1400), VAL_SAMPLES (8000).
Output checkpoint -> simulations/gantry_subnet/diagnostics/checkpoints/gantry_drift_last.pth
"""
import os
import sys
import dataclasses

import numpy as np
import torch
import deepSI

GANTRY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, GANTRY)

from gantry_dynamic.config import RunConfig
from gantry_dynamic.data import load_datasets, compute_normalization
from gantry_dynamic.model import build_model, train_model
from gantry_dynamic.training import _install_nf_val_probe
from model_augmentation.fit_systems.blocks import Static_ANN_Block

EPOCHS      = int(os.environ.get('EPOCHS', '20'))
LR          = float(os.environ.get('LR', '1.49e-8'))    # Trial 3 (the config that moved)
NF          = int(os.environ.get('NF', '1400'))
VAL_SAMPLES = int(os.environ.get('VAL_SAMPLES', '8000'))

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       '..', '..', '..', 'simulations', 'gantry_subnet',
                                       'diagnostics', 'checkpoints'))
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, 'gantry_drift_last.pth')

# ── Config: mirror the Optuna regime (X+Theta+Y, stride=100), Trial 3 lr/nf ───
cfg = RunConfig(
    ann_route_ix=(0, 1, 2, 3, 4, 5, 6, 7),
    stride=100,
    lr=LR,
    nf_override=NF,
    epochs=EPOCHS,
)
print(f'Config: routing={cfg.ann_route_ix}  stride={cfg.stride}  lr={cfg.lr:.3e}  '
      f'nf={cfg.nf}  epochs={cfg.epochs}')

np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)

# Fast validation for training (we only need _last; validation just drives _best/pruning).
_v0 = data.val_list[0]
data.val_ckpt_data = deepSI.System_data(
    u=_v0.u[:VAL_SAMPLES], y=_v0.y[:VAL_SAMPLES], dt=_v0.dt)
_t0 = data.train_list[0]
search_train = deepSI.System_data(
    u=_t0.u[:VAL_SAMPLES], y=_t0.y[:VAL_SAMPLES], dt=_t0.dt)
print(f'Cropped val for training: {data.val_ckpt_data.y.shape[0]} samples')

# ── Build + train (Trial-3 init seed), with the mandatory per-epoch nf-probe ──
trial_seed = cfg.seed + 3   # mirror Trial 3's init
np.random.seed(trial_seed); torch.manual_seed(trial_seed)
fit_sys = build_model(cfg.hp, cfg, data, norm)

_orig_cve = _install_nf_val_probe(fit_sys, cfg.hp, cfg, search_train, data.val_ckpt_data)
try:
    train_model(fit_sys, cfg.hp, cfg, data, epochs=cfg.epochs)
finally:
    fit_sys.cal_validation_error = _orig_cve

# ── Recover the DRIFTED _last (fit reloaded _best=epoch 0 at the end) ─────────
fit_sys.checkpoint_load_system(name='_last')   # in-memory model is now the drifted _last
print(f'\nReloaded _last (drifted) model. bestfit(_best)={fit_sys.bestfit:.6e}')

# ── Sanity: the ANN must be nonzero (zero-init only moves if training happened) ─
ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
ann_param_absum = sum(float(p.abs().sum()) for p in ann.net.parameters())
print(f'ANN net |params| sum = {ann_param_absum:.3e}  '
      f'({"nonzero -> trained, usable for d6" if ann_param_absum > 0 else "ZERO -> not trained!"})')

# ── Save the whole __dict__ to an explicit path (same op as checkpoint_save_system) ─
torch.save(fit_sys.__dict__, OUT_PATH)
print(f'\nSaved drift checkpoint: {OUT_PATH}')
print('Run d6 on it:')
print(f'  CKPT="{OUT_PATH}" conda run -n GraduationProject '
      f'python scripts/gantry/diagnostics-drift/d6_ann_mean_force.py')
