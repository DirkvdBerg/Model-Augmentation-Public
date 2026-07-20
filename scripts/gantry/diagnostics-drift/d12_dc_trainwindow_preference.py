"""
d12_dc_trainwindow_preference.py -- the d11 discriminator: is the trained ANN's
dY-DC PREFERRED or NEUTRAL for the windowed loss on the TRAINING distribution?

d8 measured a preference on V1 windows (paired -2.0/-2.2 SE) -- but V1 turned out
to carry a realized encoder ramp the TRAINING set does not have (d11: train-set
mean dY init bias ~ 0 at both na=17 and na=27). The training loss never saw V1's
windows, so d8's preference does not establish a training-time reward. This runs
the SAME paired full-vs-debiased windowed test on TRAINING windows:

  NEUTRAL  (|pooled Delta/SE| < 2) -> the DC is a data-silent direction on the
      training distribution: training wanders into it unopposed (shared-net
      byproduct); Layer-2 projection targets a genuinely silent direction.
  PREFERRED (Delta significantly < 0) -> a real training-time reward exists that
      is still uncharacterized; do NOT build Layer 2 yet.

Machinery = d8 (ANN shadow + fixed-mean subtraction + encoder-re-init windows),
object = gantry_drift_last.pth (na=17 trained drifted checkpoint, the one d6/d8/d9
characterized). mean_w is re-measured on THESE train windows (near-truth passes);
the ANN output is ~constant so it will be close to d8's V1 value -- reported.
4 trajectories spanning the excitation classes, ~30 windows each, paired stats
pooled (n~120, d8-level power).

Run:
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d12_dc_trainwindow_preference.py
Env: CKPT (default the d6/d8/d9 checkpoint), TRAJS (default "T3,T7,T10,T13"),
     N_WINDOWS (per trajectory, default 30), NF (default 400).
Outputs -> simulations/gantry_subnet/diagnostics/ (npz; table to stdout)
"""
import os
import sys
import time

import numpy as np
import torch

REPO   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
sys.path.insert(0, GANTRY)
sys.path.insert(0, os.path.dirname(__file__))

import deepSI
import drift_common as dc
from gantry_dynamic.data import load_datasets, compute_normalization, TRAIN_FILES
from gantry_dynamic.model import build_model, get_encoder_dims
from model_augmentation.fit_systems.blocks import Static_ANN_Block
from gantry_interconnect_dynamic import CFG as cfg

CKPT = os.environ.get(
    'CKPT', os.path.join(dc.OUT_DIR, 'checkpoints', 'gantry_drift_last.pth'))
TRAJ_TAGS = os.environ.get('TRAJS', 'T3,T7,T10,T13').split(',')
N_WINDOWS = int(os.environ.get('N_WINDOWS', '30'))
NF        = int(os.environ.get('NF', '400'))

# -- Pipeline + checkpoint (d8 pattern) -----------------------------------------
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
fit_sys = build_model(cfg.hp, cfg, data, norm)
fit_sys.__dict__ = torch.load(CKPT, map_location='cpu', weights_only=False)
print(f'Loaded CKPT {os.path.basename(CKPT)}; bestfit={getattr(fit_sys, "bestfit", float("nan")):.3e}')
fit_sys.hfn.eval()

na, nb, _, _ = get_encoder_dims(cfg.hp, cfg)
warm = max(na, nb)
ts   = cfg.ts_new
route_ix = np.asarray(cfg.ann_route_ix)
names8   = np.array(['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a'])

# selected training trajectories (the EXACT objects training consumes, incl. noise)
tag2ix = {f'T{i+1}': i for i in range(len(TRAIN_FILES))}
trajs = {t: data.train_list[tag2ix[t]] for t in TRAJ_TAGS}
print(f'Train windows: {TRAJ_TAGS} x {N_WINDOWS} windows @ nf={NF} ({NF*ts:.2f}s)')

# -- ANN shadow (d8 pattern) ------------------------------------------------------
ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
_orig_forward = ann.forward
_records  = []
_subtract = {'vec': None}


def _shadow_forward(z):
    w = _orig_forward(z)
    _records.append(w.detach().view(-1, ann.nw).cpu().numpy())
    if _subtract['vec'] is not None:
        w = w - _subtract['vec'].view(1, ann.nw, 1)
    return w


ann.forward = _shadow_forward


def _starts(tr):
    N = len(tr.u)
    st = list(range(warm, N - NF, NF))
    step = max(1, len(st) // N_WINDOWS)
    return st[::step][:N_WINDOWS]


def _windowed_rms(tr, capture=False):
    """Per-window pooled and per-channel RMS over encoder-re-init windows."""
    u = np.asarray(tr.u, dtype=np.float64)
    y = np.asarray(tr.y, dtype=np.float64)
    rms_pool, rms_ch = [], []
    if capture:
        _records.clear()
    for s in _starts(tr):
        sl = slice(s - warm, s + NF)
        sd = deepSI.System_data(u=u[sl], y=y[sl], dt=ts)
        yhat = np.asarray(fit_sys.apply_experiment(sd).y, dtype=np.float64)[-NF:]
        err = yhat - y[s:s + NF]
        rms_pool.append(np.sqrt((err ** 2).mean()))
        rms_ch.append(np.sqrt((err ** 2).mean(axis=0)))
    out = (np.asarray(rms_pool), np.asarray(rms_ch))
    if capture:
        ann_out = np.concatenate(_records, axis=0) if _records else np.zeros((0, ann.nw))
        return out + (ann_out,)
    return out


# -- Step 1: full passes, capture -> mean_w on the TRAINING distribution ---------
print('\nStep 1: full-ANN passes (capture ANN output) ...')
t0 = time.time()
full = {}
ann_all = []
for tag, tr in trajs.items():
    rp, rc, ao = _windowed_rms(tr, capture=True)
    full[tag] = (rp, rc)
    ann_all.append(ao)
    print(f'  {tag}: {len(rp)} windows  ({time.time()-t0:.0f}s cum)')
ann_all = np.concatenate(ann_all, axis=0)
mean_w = ann_all.mean(axis=0)
rms_w  = np.sqrt((ann_all ** 2).mean(axis=0))
j_dY = int(np.where(route_ix == 5)[0][0])
print(f'\n  mean_w on TRAIN windows: dY row mean={mean_w[j_dY]:.3e} [norm], '
      f'|mean|/rms={abs(mean_w[j_dY])/(rms_w[j_dY]+1e-30):.3f}  '
      f'(d8/V1 was -1.413e-06, 0.997)')

# -- Step 2: debiased passes ------------------------------------------------------
print('\nStep 2: debiased passes (ANN minus TRAIN mean_w) ...')
t0 = time.time()
_subtract['vec'] = torch.tensor(mean_w, dtype=next(ann.parameters()).dtype)
deb = {}
for tag, tr in trajs.items():
    rp, rc = _windowed_rms(tr)
    deb[tag] = (rp, rc)
    print(f'  {tag}: done  ({time.time()-t0:.0f}s cum)')
_subtract['vec'] = None

# -- Paired statistics -------------------------------------------------------------
print('\n=== Paired Delta = RMS(full) - RMS(debiased) on TRAINING windows ===')
print('    (NEGATIVE = the DC-carrying model fits the window BETTER = preference)')
print(f"  {'traj':6s} {'n':>4s} {'Delta pooled':>13s} {'/SE':>7s} {'Delta Y-ch':>12s} {'/SE':>7s}")
d_pool_all, d_y_all = [], []
for tag in trajs:
    d_pool = full[tag][0] - deb[tag][0]
    d_y    = full[tag][1][:, 2] - deb[tag][1][:, 2]
    d_pool_all.append(d_pool); d_y_all.append(d_y)
    sp = d_pool.std(ddof=1) / np.sqrt(len(d_pool))
    sy = d_y.std(ddof=1) / np.sqrt(len(d_y))
    print(f'  {tag:6s} {len(d_pool):>4d} {d_pool.mean():>13.3e} {d_pool.mean()/sp:>7.2f} '
          f'{d_y.mean():>12.3e} {d_y.mean()/sy:>7.2f}')
d_pool_all = np.concatenate(d_pool_all); d_y_all = np.concatenate(d_y_all)
sp = d_pool_all.std(ddof=1) / np.sqrt(len(d_pool_all))
sy = d_y_all.std(ddof=1) / np.sqrt(len(d_y_all))
print(f'  {"POOLED":6s} {len(d_pool_all):>4d} {d_pool_all.mean():>13.3e} '
      f'{d_pool_all.mean()/sp:>7.2f} {d_y_all.mean():>12.3e} {d_y_all.mean()/sy:>7.2f}')
print(f'\n  d8/V1 reference: pooled Delta/SE = -2.0, Y Delta/SE = -2.2 (preference).')
print('  |pooled /SE| < 2 here -> NEUTRAL on the training distribution -> the DC is')
print('  data-silent for training (Layer-2 premise holds); significantly negative ->')
print('  a real training-time reward exists, keep hunting before building Layer 2.')

stem = os.path.join(dc.OUT_DIR, 'd12_dc_trainwindow_preference')
np.savez(stem + '.npz', mean_w=mean_w, rms_w=rms_w, route_ix=route_ix,
         d_pool=d_pool_all, d_y=d_y_all, trajs=np.array(TRAJ_TAGS),
         n_windows=N_WINDOWS, nf=NF, ts=ts, ckpt=str(CKPT),
         **{f'full_pool_{t}': full[t][0] for t in trajs},
         **{f'deb_pool_{t}': deb[t][0] for t in trajs},
         **{f'full_ch_{t}': full[t][1] for t in trajs},
         **{f'deb_ch_{t}': deb[t][1] for t in trajs})
print(f'\nSaved: {stem}.npz')
