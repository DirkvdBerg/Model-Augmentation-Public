"""
d16_pilow_spectrum.py -- the LAST pre-implementation Layer-2 diagnostic: compute
the actual Pi_low object offline (loss-gradient Gram spectrum over a probe basis)
and locate the measured DC direction and the absorber-band directions in it.

Layer 2 needs a target subspace Pi_low computed from data. The candidate
construction (concept doc §3) is the SVD/eigen-structure of the empirical
loss-sensitivity Gram. Its declared failure mode (limits C1) is MISALIGNMENT:
the spectrum-based low-information subspace might not contain the drift
direction. This diagnostic builds the Gram for the REAL pipeline and answers,
before any implementation:

  Q1  where does the measured DC/drift direction (mean_w) rank in the spectrum?
  Q2  where do the absorber-band directions rank?
  Q3  does a DATA-DERIVED cutoff exist (spectral gap) such that Pi_low contains
      the drift direction and excludes the band directions?
      YES -> the Fisher/Gram construction is viable as-is (C1 de-risked).
      NO  -> use the DIRECT measured-DC pin (or frequency-weighted variant)
             instead of a pure information criterion; the Gram is then only
             the safety check, not the target constructor.

Method (no training, no pipeline change; frozen trained checkpoint):
  Probe basis (np = 12 amplitudes): DC on each of the 8 routed rows; sin/cos at
  f_a = 150 Hz on the dY row and the dTheta row. Per training window i, the
  gradient g_i in R^12 of the window's pooled RMS w.r.t. the probe amplitudes,
  by CENTRAL finite differences at eps = 1e-6 [norm] (operational scale, d14;
  responses verified linear in d13/d14). Gram G = (1/n) sum g_i g_i^T; its
  eigen-decomposition is the Pi_low constructor. Rayleigh quotients locate the
  named directions in the spectrum.

Cost basis: (1 + 2*12) passes x per-pass time measured in d14 (~55 s at 40
windows) -- run the full version in the background.

Run:
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d16_pilow_spectrum.py
Env: CKPT (default gantry_drift_last.pth), TRAJS ("T3,T10"), N_WINDOWS (20),
     NF (400), EPS (1e-6).
Outputs -> simulations/gantry_subnet/diagnostics/ (npz + png)
"""
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
TRAJ_TAGS = os.environ.get('TRAJS', 'T3,T10').split(',')
N_WINDOWS = int(os.environ.get('N_WINDOWS', '20'))
NF        = int(os.environ.get('NF', '400'))
EPS       = float(os.environ.get('EPS', '1e-6'))
FA        = dc.fa

# ── Pipeline + trained checkpoint (d14 pattern) ───────────────────────────────
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
names8 = np.array(['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a'])
tag2ix = {f'T{i+1}': i for i in range(len(TRAIN_FILES))}
trajs = {t: data.train_list[tag2ix[t]] for t in TRAJ_TAGS}
nw = len(route_ix)
j = {int(r): int(np.where(route_ix == r)[0][0]) for r in route_ix}

# ── Probe basis: (vec in R^nw, freq, phase) per probe amplitude ───────────────
def _unit(ix):
    v = np.zeros(nw); v[j[ix]] = 1.0
    return v

PROBES = [(f'DC {names8[r]}', _unit(r), 0.0, 0.0) for r in route_ix]
PROBES += [
    ('sin150 dY',     _unit(5), FA, 0.0),
    ('cos150 dY',     _unit(5), FA, np.pi / 2),
    ('sin150 dTheta', _unit(4), FA, 0.0),
    ('cos150 dTheta', _unit(4), FA, np.pi / 2),
]
NP_ = len(PROBES)
plabels = [p[0] for p in PROBES]

# ── ANN shadow (d14 pattern) ──────────────────────────────────────────────────
ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
_orig_forward = ann.forward
_probe = {'vec': None, 'freq': 0.0, 'amp': 0.0, 'phase': 0.0, 'k': 0}


def _shadow_forward(z):
    w = _orig_forward(z)
    if _probe['vec'] is not None:
        s = 1.0 if _probe['freq'] == 0.0 else float(
            np.sin(2 * np.pi * _probe['freq'] * _probe['k'] * ts + _probe['phase']))
        w = w + (_probe['amp'] * s) * _probe['vec'].view(1, ann.nw, 1)
    _probe['k'] += 1
    return w


ann.forward = _shadow_forward


def _starts(tr):
    N = len(tr.u)
    st = list(range(warm, N - NF, NF))
    step = max(1, len(st) // N_WINDOWS)
    return st[::step][:N_WINDOWS]


def _windowed_rms(vec=None, freq=0.0, amp=0.0, phase=0.0):
    _probe.update(vec=vec, freq=freq, amp=amp, phase=phase)
    pooled = []
    for tag, tr in trajs.items():
        u = np.asarray(tr.u, dtype=np.float64)
        y = np.asarray(tr.y, dtype=np.float64)
        for s0 in _starts(tr):
            _probe['k'] = 0
            sl = slice(s0 - warm, s0 + NF)
            sd = deepSI.System_data(u=u[sl], y=y[sl], dt=ts)
            yhat = np.asarray(fit_sys.apply_experiment(sd).y, dtype=np.float64)[-NF:]
            pooled.append(np.sqrt(((yhat - y[s0:s0 + NF]) ** 2).mean()))
    _probe['vec'] = None
    return np.asarray(pooled)


# ── Per-window gradients by central differences ───────────────────────────────
print(f'\nReference pass ({TRAJ_TAGS} x {N_WINDOWS} @ nf={NF}, eps={EPS:.1e}) ...')
t0 = time.time()
rms_ref = _windowed_rms()
nwin = len(rms_ref)
print(f'  {nwin} windows, ref pooled RMS = {rms_ref.mean():.6e}  ({time.time()-t0:.0f}s)')

dt_ = next(ann.parameters()).dtype
Gmat = np.empty((nwin, NP_))                    # per-window gradients
for p, (name, v, freq, phase) in enumerate(PROBES):
    vt = torch.tensor(v, dtype=dt_)
    t0 = time.time()
    rp = _windowed_rms(vec=vt, freq=freq, amp=+EPS, phase=phase)
    rm = _windowed_rms(vec=vt, freq=freq, amp=-EPS, phase=phase)
    Gmat[:, p] = (rp - rm) / (2 * EPS)
    print(f'  grad {p+1:2d}/{NP_} {name:16s} |mean g|={abs(Gmat[:, p].mean()):.3e} '
          f'rms g={np.sqrt((Gmat[:, p]**2).mean()):.3e}   ({time.time()-t0:.0f}s)')

# ── Gram, spectrum, direction placement ───────────────────────────────────────
G = (Gmat.T @ Gmat) / nwin                       # (NP_, NP_)
evals, evecs = np.linalg.eigh(G)                 # ascending
evals = evals[::-1]; evecs = evecs[:, ::-1]      # descending

print('\n=== Gram spectrum (information per probe direction, descending) ===')
for i, ev in enumerate(evals):
    top = np.argsort(-np.abs(evecs[:, i]))[:3]
    comp = ', '.join(f'{plabels[t]} {evecs[t, i]:+.2f}' for t in top)
    print(f'  ev{i:2d}: {ev:.3e}   [{comp}]')


def rayleigh(vec12):
    v = vec12 / (np.linalg.norm(vec12) + 1e-30)
    return float(v @ G @ v)


# named directions in probe space
mean_w_probe = np.zeros(NP_)
z6 = np.load(os.path.join(dc.OUT_DIR, 'd6_ann_mean_force_gantry_drift_last.npz'))
mw = z6['mean_w']
mean_w_probe[:nw] = mw / (np.linalg.norm(mw) + 1e-30)
named = {
    'mean_w (drift direction)': mean_w_probe,
    'DC dY only': np.eye(NP_)[j[5]],
    'band dY (sin)': np.eye(NP_)[nw + 0],
    'band dY (cos)': np.eye(NP_)[nw + 1],
    'band dTheta (sin)': np.eye(NP_)[nw + 2],
    'band dTheta (cos)': np.eye(NP_)[nw + 3],
}
print('\n=== Named directions: Rayleigh quotient + spectrum percentile ===')
ray = {}
for name, v in named.items():
    r = rayleigh(v)
    pct = float((evals < r).mean() * 100)        # % of eigenvalues below it
    ray[name] = (r, pct)
    print(f'  {name:28s} R = {r:.3e}   above {pct:4.0f}% of the spectrum')

# ── Q3: does a separating cutoff exist? ───────────────────────────────────────
r_drift = ray['mean_w (drift direction)'][0]
r_band_min = min(ray[k][0] for k in ray if k.startswith('band'))
print(f'\nQ3: drift-direction R = {r_drift:.3e}  vs  min band R = {r_band_min:.3e}  '
      f'(ratio band/drift = {r_band_min / (r_drift + 1e-30):.2f})')
if r_band_min / (r_drift + 1e-30) > 10:
    print('-> a separating cutoff EXISTS: an information-spectrum Pi_low can pin the')
    print('   drift direction while sparing the band (C1 de-risked; Fisher/Gram viable).')
elif r_band_min > r_drift:
    print('-> band ranks above drift but by <10x: a pure spectrum cutoff is FRAGILE;')
    print('   prefer the DIRECT measured-DC pin, keep the Gram as the safety check.')
else:
    print('-> MISALIGNED (C1 confirmed): the loss carries MORE information about the')
    print('   drift direction than about the band at probe scale; a pure information')
    print('   cutoff will NOT pin the drift. Build the DIRECT measured-DC pin.')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].semilogy(np.arange(NP_), evals, 'o-')
for name, (r, _) in ray.items():
    axes[0].axhline(r, ls=':', lw=1,
                    color='C3' if 'drift' in name or 'dY only' in name else 'C2',
                    label=name)
axes[0].set_xlabel('eigenvalue index (descending)')
axes[0].set_ylabel('Gram eigenvalue / Rayleigh quotient')
axes[0].set_title('Information spectrum + named directions')
axes[0].grid(True, which='both'); axes[0].legend(fontsize=6, loc='lower left')
im = axes[1].imshow(np.log10(np.abs(G) + 1e-30), cmap='viridis')
axes[1].set_xticks(range(NP_)); axes[1].set_xticklabels(plabels, rotation=90, fontsize=6)
axes[1].set_yticks(range(NP_)); axes[1].set_yticklabels(plabels, fontsize=6)
axes[1].set_title('log10 |Gram|')
fig.colorbar(im, ax=axes[1], shrink=0.8)
fig.suptitle(f'd16: offline Pi_low spectrum on {os.path.basename(CKPT)} '
             f'({nwin} windows, eps={EPS:.0e})')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, 'd16_pilow_spectrum')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', Gmat=Gmat, G=G, evals=evals, evecs=evecs,
         plabels=np.array(plabels), eps=EPS, rms_ref=rms_ref,
         ray_names=np.array(list(ray)), ray_vals=np.array([ray[k] for k in ray]),
         mean_w=mw, route_ix=route_ix, nf=NF, n_windows=N_WINDOWS,
         trajs=np.array(TRAJ_TAGS), ckpt=str(CKPT))
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
