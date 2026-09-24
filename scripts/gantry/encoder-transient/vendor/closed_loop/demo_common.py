"""
demo_common.py -- shared infrastructure for the drift-demo figures (plan doc §12).

Provides, with a SLIM data load (train trajectories only; no val/test/GT loads):
  - build_pipeline(): the exact pipeline encoder + normalization (build_model), minutes -> seconds
  - load_T(): one trajectory (u, y, x_logical, delta_a) from an explicit data dir
  - encoder_x0(): the untrained linear-map encoder's physical x0 for a record
  - re-exports from diagnostics-drift/drift_common: simulate_baseline, tau_X, tau_Y, P-helpers
  - figure helpers: sci notation on all y-axes (rule R-c style), provenance annotation (rule R-b),
    amp_spectrum, OUT_DIR (simulations/gantry_subnet/diagnostics/drift-demo/)

Faithfulness notes:
  - The encoder normalization is a TRAINING-set statistic, so the 14 train trajectories are loaded
    (that is the irreducible cost); val/test/aug-GT loads and the val noise are skipped.
  - If cfg.snr is set, the train outputs are noised in the SAME rng order as load_datasets
    (train first), so the normalization matches the pipeline exactly.
  - Seeding mirrors the entry file: seed before data, seed again before build_model.
"""
__project_origin__ = "added"

import os
import sys
import time

import numpy as np
import torch
import deepSI
from scipy.io import loadmat

REPO   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
sys.path.insert(0, GANTRY)                                        # gantry_dynamic + entry file
sys.path.insert(0, os.path.join(GANTRY, 'diagnostics-drift'))     # drift_common

import drift_common as dc
from gantry_dynamic.data import (DataBundle, TRAIN_FILES, load_traj, traj_dir,
                                 _load_u, _resample_u, compute_normalization)
from gantry_dynamic.model import build_model, get_encoder_dims
from gantry_dynamic.diagnostics import encoder_init_state
from gantry_interconnect_dynamic import CFG               # exact run config (no training on import)

# Re-exports (verified physics; do not re-derive)
simulate_baseline = dc.simulate_baseline
tau_X, tau_Y      = dc.tau_X, dc.tau_Y
stage_pos_to_logical = dc.stage_pos_to_logical

OUT_DIR = os.path.join(REPO, 'scripts', 'gantry', 'drift-demo', 'figures')   # user 2026-07-14
os.makedirs(OUT_DIR, exist_ok=True)

STATE_NAMES = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY']


def build_pipeline(cfg=CFG, verbose=True):
    """Slim pipeline: train-trajectory load -> normalization -> build_model (untrained).

    Returns (fit_sys, norm, K0, na, nb, na_right, nb_right). fit_sys's ANN is zero-init
    (exact zero output), so fit_sys's encoder is the pipeline encoder and the augmented
    model equals the baseline.
    """
    t0 = time.time()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    train_list = [load_traj(f, cfg) for f in TRAIN_FILES]
    if cfg.snr is not None:
        # Reproduce load_datasets' noise stream (train files are noised FIRST there).
        if   cfg.snr == 50: sigma_n = 4.5e-4
        elif cfg.snr == 55: sigma_n = 2.5e-4
        elif cfg.snr == 60: sigma_n = 1.4e-4
        else: raise ValueError('SNR must be 50, 55, 60 or None')
        for sd in train_list:
            sd.y = sd.y + np.random.normal(0, sigma_n, sd.y.shape).astype(cfg.dtype_np)
    train_data = deepSI.System_data_list(train_list)
    bundle = DataBundle(train_list=train_list, val_list=None, test_list=None,
                        train_data=train_data, val_ckpt_data=None, val_data=None,
                        test_data=None, val_x_logical=None, val_x_aug=None,
                        test_x_logical=None, test_x_aug=None, sigma_n=None)
    norm = compute_normalization(cfg, bundle)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(cfg.hp, cfg, bundle, norm)
    na, nb, na_right, nb_right = get_encoder_dims(cfg.hp, cfg)
    K0 = max(na, nb)
    if verbose:
        print(f'[demo_common] pipeline built in {time.time()-t0:.0f}s '
              f'(slim load: {len(TRAIN_FILES)} train files, snr={cfg.snr}, '
              f'na=nb={na}, K0={K0})')
    return fit_sys, norm, K0, na, nb, na_right, nb_right


def load_T(filename, cfg=CFG, subdir=None, need_absorber=False):
    """Load u (block-mean stage force), y (stage, noiseless), x_logical (N,6), delta_a
    from <traj_dir>/<subdir>/<filename> (subdir=None -> with-MSD root; 'baseline' -> no-MSD)."""
    directory = traj_dir(cfg) if subdir is None else os.path.join(traj_dir(cfg), subdir)
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = loadmat(path, squeeze_me=True)
    u = _resample_u(_load_u(d), cfg).astype(cfg.dtype_np)
    N = len(u)
    D = cfg.d
    y = d['y'][::D][:N].astype(cfg.dtype_np)
    x_logical = d['x_logical'][::D][:N].astype(cfg.dtype_np)     # (N,6) [X,Th,Y,dX,dTh,dY]
    delta_a = None
    if need_absorber and 'delta_a' in d:
        delta_a = d['delta_a'][::D][:N].astype(cfg.dtype_np)
    return u, y, x_logical, delta_a


def encoder_x0(fit_sys, norm, u, y, K0, na, nb, na_right, nb_right, cfg=CFG):
    """Untrained pipeline-encoder estimate of the PHYSICAL state at sample K0 (6,)."""
    sd = deepSI.System_data(u=u, y=y, dt=cfg.ts_new)
    x0_norm = encoder_init_state(fit_sys, sd, K0, na, nb, na_right, nb_right, cfg)
    return x0_norm * norm.std_x.flatten() + norm.x_mean.flatten()


def amp_spectrum(x, fs):
    """Single-sided amplitude spectrum (Hann-windowed)."""
    x = np.asarray(x, dtype=np.float64)
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * w)) * 2.0 / w.sum()
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return f, X


# ── Figure helpers (plan §12 rules) ───────────────────────────────────────────
def sci_axes(ax):
    """Rule: consistent scientific notation on every y-axis."""
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))


def add_provenance(fig, text):
    """Rule R-b: checkpoint/config provenance on every figure (small, bottom-left)."""
    fig.text(0.005, 0.005, text, fontsize=6, color='0.45', va='bottom')
