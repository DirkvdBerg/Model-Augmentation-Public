"""
d9_dc_compensation_shape.py -- what systematic in-window Y trend does the trained
ANN's DC compensate? (follow-up to d8 / D-109)

d8 measured: the windowed loss PREFERS the drift-driving Y-DC at every feasible
horizon (removing it makes the <=1 s windowed fit worse). Since the sim's true
residual (absorber) is zero-mean, that RMS benefit cannot be real signal -- the DC
must cancel a systematic in-window trend from elsewhere in the pipeline. This
probe identifies the systematic by SHAPE + two independent coefficient checks.

Logic (both models see the SAME systematic; only the full model carries the DC):
    e_full(t)  ~ systematic(t) + DC_effect(t)   ~ small   (that is WHY full wins)
    e_deb(t)   ~ systematic(t)                            (DC removed -> exposed)
so the ensemble-mean SIGNED Y error of the DEBIASED windowed passes is the
systematic itself, and  e_full - e_deb  ~ DC_effect (a mechanical consistency
check, predictable from the measured mean_w alone).

Hypotheses (fit e_deb_Y(t) = a + b*t + c*t^2 over the 0.1 s window):
  H1 encoder-init bias : linear, b ~ measured mean dY init error, a ~ mean Y
     init error (both measured directly against val_x_logical at each window
     start -- Measurement 2). Fix = encoder init, at the source.
  H2 force-like systematic (baseline/discretization/M(Y)) : quadratic dominant,
     c ~ -0.5 * a_DC where a_DC = (dY-row DC in m/s per step)/ts is the
     acceleration the ANN injects (the DC cancels the systematic, so the exposed
     curvature is the negative of the DC's own).
  H3 neither (flat/noise) : the d8 RMS advantage is not trend-cancellation ->
     co-training/off-manifold artifact; re-derive the Layer-2 premise.

Ensemble averaging over the non-overlapping windows suppresses the zero-mean
absorber oscillation by ~sqrt(nwin), which is what makes the slow trend readable.

Run:
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d9_dc_compensation_shape.py
Env: CKPT      (default simulations/gantry_subnet/diagnostics/checkpoints/gantry_drift_last.pth)
     NF        (window length in samples, default 400 = the training horizon)
     N_WINDOWS (cap; default all non-overlapping)
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
from gantry_dynamic.data import load_datasets, compute_normalization
from gantry_dynamic.model import build_model, get_encoder_dims
from gantry_dynamic.diagnostics import encoder_init_state
from model_augmentation.fit_systems.blocks import Static_ANN_Block
from gantry_interconnect_dynamic import CFG as cfg

CKPT = os.environ.get(
    'CKPT', os.path.join(dc.OUT_DIR, 'checkpoints', 'gantry_drift_last.pth'))
NF        = int(os.environ.get('NF', '400'))
N_WINDOWS = os.environ.get('N_WINDOWS', None)

# -- Build pipeline + load the trained drifted checkpoint (d8 pattern) ---------
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
fit_sys = build_model(cfg.hp, cfg, data, norm)
if CKPT and os.path.exists(CKPT):
    fit_sys.__dict__ = torch.load(CKPT, map_location='cpu', weights_only=False)
    print(f'Loaded CKPT {os.path.basename(CKPT)}; bestfit={getattr(fit_sys, "bestfit", float("nan")):.3e}')
else:
    print(f'WARNING: CKPT not found ({CKPT}); untrained ANN -- machinery check only.')
    CKPT = None
fit_sys.hfn.eval()

na, nb, na_right, nb_right = get_encoder_dims(cfg.hp, cfg)
warm = max(na, nb)
ts   = cfg.ts_new
route_ix = list(cfg.ann_route_ix)
std_x    = norm.std_x.flatten()

v1 = data.val_data
u  = np.asarray(v1.u, dtype=np.float64)
y  = np.asarray(v1.y, dtype=np.float64)
x_true = data.val_x_logical.astype(np.float64)          # (N, 6) ground truth
Ntot = len(u)
absorber_rms = data.val_x_aug[:, 0].std()

starts = list(range(warm, Ntot - NF, NF))
if N_WINDOWS is not None:
    step = max(1, len(starts) // int(N_WINDOWS))
    starts = starts[::step][:int(N_WINDOWS)]
print(f'V1: {Ntot} samples @ {1/ts:.0f} Hz  warm={warm}  NF={NF} ({NF*ts:.2f}s)  '
      f'{len(starts)} windows  absorber RMS={absorber_rms:.3e} m')

# -- Shadow the ANN: record + optional fixed-vector subtraction (d8 pattern) ---
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


def _windowed_signed_errors(capture=False):
    """(nwin, NF, 3) SIGNED errors, encoder re-init per window (d7/d8 geometry)."""
    errs = []
    if capture:
        _records.clear()
    for s in starts:
        sl = slice(s - warm, s + NF)
        sd = deepSI.System_data(u=u[sl], y=y[sl], dt=ts)
        yhat = np.asarray(fit_sys.apply_experiment(sd).y, dtype=np.float64)[-NF:]
        errs.append(yhat - y[s:s + NF])
    out = np.asarray(errs)
    if capture:
        ann_out = np.concatenate(_records, axis=0) if _records else np.zeros((0, ann.nw))
        return out, ann_out
    return out


# -- Measurement 2 (cheap, first): encoder init error at each window start -----
# encoder_init_state is the minimal encoder call (no simulation).
print('\nMeasurement 2: encoder init error at each window start ...')
init_err = np.empty((len(starts), 6))
for i, s in enumerate(starts):
    x0n = encoder_init_state(fit_sys, v1, s, na, nb, na_right, nb_right, cfg)
    x0  = x0n * std_x + norm.x_mean.flatten()
    init_err[i] = x0 - x_true[s]
names6 = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY']
mean_ie = init_err.mean(axis=0)
se_ie   = init_err.std(axis=0, ddof=1) / np.sqrt(len(starts))
print(f"  {'state':8s} {'mean err':>12s} {'SE':>10s} {'mean/SE':>8s}")
for i, nm in enumerate(names6):
    r = mean_ie[i] / se_ie[i] if se_ie[i] > 0 else np.nan
    print(f'  {nm:8s} {mean_ie[i]:>12.3e} {se_ie[i]:>10.2e} {r:>8.2f}')

# -- Pass A: FULL model, capture ANN output -> the DC's own effect prediction --
print('\nPass A: full ANN windowed pass (signed errors + ANN capture) ...')
t0 = time.time()
err_full, ann_out = _windowed_signed_errors(capture=True)
print(f'  done ({time.time()-t0:.0f}s)')
mean_w = ann_out.mean(axis=0)

# DC composition -> predicted in-window effect on Y (logical Y == output ch 2):
# a per-step increment m on state row r acts once per output step:
#   position row (idx 2): velocity-like slope  v_DC = m_phys / ts       [m/s]
#   velocity row (idx 5): acceleration-like    a_DC = m_phys / ts       [m/s^2]
# THEORY: double/single integration of a constant per-step state increment; the
# DC effect on position over a window is v_DC*t + 0.5*a_DC*t^2.
v_DC = a_DC = 0.0
for j, r in enumerate(route_ix):
    if r == 2:
        v_DC = float(mean_w[j] * std_x[2] / ts)
    elif r == 5:
        a_DC = float(mean_w[j] * std_x[5] / ts)
print(f'\n  DC composition (measured mean_w): v_DC={v_DC:+.3e} m/s (Y row), '
      f'a_DC={a_DC:+.3e} m/s^2 (dY row)')
print(f'  predicted DC effect at t={NF*ts:.2f}s: '
      f'{v_DC*NF*ts + 0.5*a_DC*(NF*ts)**2:+.3e} m')

# -- Pass B: DEBIASED model (ANN minus mean_w) ----------------------------------
print('\nPass B: debiased windowed pass (signed errors) ...')
t0 = time.time()
_subtract['vec'] = torch.tensor(mean_w, dtype=next(ann.parameters()).dtype)
err_deb = _windowed_signed_errors()
_subtract['vec'] = None
print(f'  done ({time.time()-t0:.0f}s)')

# -- Ensemble means + quadratic fit on the exposed systematic -------------------
t = np.arange(NF) * ts
e_full = err_full.mean(axis=0)                 # (NF, 3) signed ensemble mean
e_deb  = err_deb.mean(axis=0)
se_deb = err_deb.std(axis=0, ddof=1) / np.sqrt(len(starts))

A = np.vstack([np.ones_like(t), t, t ** 2]).T   # [a, b, c] design matrix


def _fit(ecurve):
    coef, *_ = np.linalg.lstsq(A, ecurve, rcond=None)
    resid = ecurve - A @ coef
    return coef, np.sqrt((resid ** 2).mean())


print('\n=== Quadratic fit  e(t) = a + b*t + c*t^2  on the ensemble-mean Y error ===')
rows = {}
for lbl, e in [('full', e_full[:, 2]), ('debiased', e_deb[:, 2]),
               ('exposed (deb - full)', e_deb[:, 2] - e_full[:, 2])]:
    (aa, bb, cc), rr = _fit(e)
    rows[lbl] = (aa, bb, cc, rr)
    print(f'  {lbl:22s} a={aa:+.3e} m  b={bb:+.3e} m/s  c={cc:+.3e} m/s^2  (fit resid RMS {rr:.1e})')

aa, bb, cc, _ = rows['debiased']
T = NF * ts
lin_contrib  = abs(bb) * T
quad_contrib = abs(cc) * T ** 2
print(f'\n  debiased-trend contributions at t={T:.2f}s: |b|*T={lin_contrib:.3e} m  '
      f'|c|*T^2={quad_contrib:.3e} m')

print('\n=== Independent coefficient checks ===')
print(f'  H1 (encoder bias):  fitted a={aa:+.3e} vs mean Y  init err {mean_ie[2]:+.3e} m')
print(f'                      fitted b={bb:+.3e} vs mean dY init err {mean_ie[5]:+.3e} m/s')
print(f'  H2 (force-like)  :  fitted c={cc:+.3e} vs -0.5*a_DC = {-0.5*a_DC:+.3e} m/s^2')
print(f'  machinery check  :  exposed-trend coefs should mirror the DC effect: '
      f'b~{-v_DC:+.3e}, c~{-0.5*a_DC:+.3e}')

# significance of the trend against the ensemble SE at window end
sig = abs(e_deb[-1, 2]) / (se_deb[-1, 2] + 1e-30)
print(f'\n  debiased mean Y error at window end: {e_deb[-1,2]:+.3e} m '
      f'(SE {se_deb[-1,2]:.2e}, |mean|/SE = {sig:.1f})')
if sig < 2:
    print('  !! trend not significant vs ensemble SE -> leans H3 (no clean systematic).')

# -- Plot ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
chan = ['X1', 'X2', 'Y']
for ch, ax in enumerate(axes):
    ax.plot(t * 1e3, e_full[:, ch] * 1e6, 'C0', lw=1.2, label='full ANN (has DC)')
    ax.plot(t * 1e3, e_deb[:, ch] * 1e6, 'C3', lw=1.2, label='debiased (DC removed)')
    ax.fill_between(t * 1e3, (e_deb[:, ch] - 2 * se_deb[:, ch]) * 1e6,
                    (e_deb[:, ch] + 2 * se_deb[:, ch]) * 1e6, color='C3', alpha=0.15,
                    label='debiased +/- 2 SE' if ch == 0 else None)
    if ch == 2:
        afit, bfit, cfit, _ = rows['debiased']
        ax.plot(t * 1e3, (afit + bfit * t + cfit * t ** 2) * 1e6, 'k--', lw=1.0,
                label='quadratic fit (debiased)')
        pred = mean_ie[2] + mean_ie[5] * t
        ax.plot(t * 1e3, pred * 1e6, 'g:', lw=1.5,
                label='H1 pred: Y0err + dY0err*t')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('time into window [ms]'); ax.set_title(chan[ch])
    ax.grid(True); ax.legend(fontsize=7)
axes[0].set_ylabel('ensemble-mean signed error [um]')
tag = os.path.basename(CKPT).split('.')[0] if CKPT else 'untrained'
fig.suptitle(f'd9: which systematic does the Y-DC compensate?  ({tag}, {len(starts)} windows, NF={NF})')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, f'd9_dc_compensation_{tag}')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', t=t, e_full=e_full, e_deb=e_deb, se_deb=se_deb,
         err_full=err_full, err_deb=err_deb, init_err=init_err, mean_ie=mean_ie,
         se_ie=se_ie, mean_w=mean_w, v_DC=v_DC, a_DC=a_DC,
         fit_full=np.array(rows['full'][:3]), fit_deb=np.array(rows['debiased'][:3]),
         fit_exposed=np.array(rows['exposed (deb - full)'][:3]),
         starts=np.array(starts), NF=NF, ts=ts, absorber_rms=absorber_rms, ckpt=str(CKPT))
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
