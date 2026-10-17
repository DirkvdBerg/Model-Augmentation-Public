"""Tests of the least-squares encoder start (D-271, tasks/encoder-lsq-proposal-2026-10-17.md), on the production code.

T1 (local, default): 2 s of every record, CPU. Through the pipeline's own build_model / build_closed_loop /
  prefit_physical_encoder (which checks finiteness, element-wise reproduction, R^2 >= 0.999 per state and unchanged
  W^a and added rows itself):
  1. (detuned, seed 1, n_a 2) twice on fresh models: identical W^b and intercept;
  2. n_a = 0 and 8 build and start;
  3. the readiness check stops a start whose targets the windows cannot explain (random targets).
T2 (server, ENC_T2=<0|1|2>): full data on the run's device, one distinct pre-fit problem of the final campaign:
  0 = (detuned, seed 1): R1, R4, R13, R14;  1 = (correct, seed 1): R2, R3, R5, R6, R7;  2 = (detuned, seed 2).
  Prints the per-state line and a hash of the start; pass = no error and equal hashes for the two repeats.
"""
import hashlib
import os
import sys
from dataclasses import replace

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))

T2_PROBLEMS = [('detuned', 1), ('correct', 1), ('detuned', 2)]
t2 = os.environ.get('ENC_T2')
os.environ.update(THESIS_ARM='u', THESIS_NOISE='lowpass', THESIS_START='detuned', THESIS_NA='2', THESIS_SEED='1',
                  THESIS_PS2='1', THESIS_INIT='live')
if t2 is None:
    os.environ.update(THESIS_SMOKE='1', THESIS_DEVICE='cpu')

import gantry_interconnect_dynamic as gid                                       # noqa: E402
from gantry_dynamic.data import load_datasets, compute_normalization            # noqa: E402
from gantry_dynamic.model import build_model, prefit_physical_encoder, least_squares_start   # noqa: E402
from gantry_dynamic.controller import build_closed_loop                         # noqa: E402

_data = {}


def config(start, seed, na):
    """The THESIS block's configuration for another start, seed or n_a (same rules as gantry_interconnect_dynamic)."""
    detune = None
    if start == 'detuned':
        detune = [float(1.0 + 0.10 * s) for s in np.random.default_rng(seed).choice([-1.0, 1.0], size=10)]
    return replace(gid.CFG, seed=seed, combo_init_detune=detune, nx_ann=na, ann_route_ix=tuple(range(6 + na)))


def start(cfg):
    """Build the run's model as main() does and set its encoder start; return the model and a hash of W^b + bias."""
    if cfg.mode not in _data:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        data = load_datasets(cfg)
        _data[cfg.mode] = (data, compute_normalization(cfg, data))
    data, norm = _data[cfg.mode]
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(cfg.hp, cfg, data, norm)
    fit_sys.simulator = build_closed_loop(fit_sys, norm, cfg, train_files=data.train_files, val_files=data.val_files,
                                          val_data=data.val_ckpt_data, e_train=data.e_train, e_val=data.e_val)
    prefit_physical_encoder(fit_sys, cfg, data, norm)
    enc = fit_sys.encoder
    h = hashlib.sha256()
    for t in (enc.Wb_psi_y, enc.Wb_psi_u, enc.net[-1].bias[:cfg.nx_phys]):
        h.update(t.detach().cpu().numpy().tobytes())
    return fit_sys, h.hexdigest()[:16]


def t1():
    cfg = config('detuned', 1, 2)
    _, h1 = start(cfg)
    _, h2 = start(cfg)
    print('[T1] 1 determinism: %s %s -> %s' % (h1, h2, 'PASS' if h1 == h2 else 'FAIL'), flush=True)
    ok = h1 == h2
    for na in (0, 8):
        _, h = start(config('detuned', 1, na))
        print('[T1] 2 n_a %d: start set (%s) PASS' % (na, h), flush=True)
    fit_sys, _ = start(cfg)
    g = torch.Generator().manual_seed(0)
    uh = torch.randn(4000, fit_sys.nb, cfg.nu, generator=g, dtype=torch.float64)
    yh = torch.randn(4000, fit_sys.na, cfg.ny, generator=g, dtype=torch.float64)
    xt = torch.randn(4000, cfg.nx_phys, generator=g, dtype=torch.float64)
    try:
        least_squares_start(fit_sys.encoder, cfg.nx_phys, uh, yh, xt)
        print('[T1] 3 readiness check on unexplainable targets: did not stop -> FAIL', flush=True)
        ok = False
    except RuntimeError as e:
        stopped = 'not ready' in str(e)
        print('[T1] 3 readiness check on unexplainable targets: %s -> %s' % (e, 'PASS' if stopped else 'FAIL'),
              flush=True)
        ok = ok and stopped
    print('[T1] ALL PASS' if ok else '[T1] FAIL', flush=True)
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    if t2 is None:
        t1()
    else:
        s, seed = T2_PROBLEMS[int(t2)]
        _, h = start(config(s, seed, 2))
        print('[T2] problem %s (%s, seed %d): start set, hash %s' % (t2, s, seed, h), flush=True)
