"""
dB_boundedintegral_projection.py -- D-B: does the bounded-integral (high-pass) projection remove the
drift on the trained checkpoint WHILE preserving the 130-180 Hz absorber capture? (no retraining)

On the drifted X+Theta+Y checkpoint, transform the ANN output on the routed rows DURING the free-run
and compare three treatments:
  raw       : ANN output unchanged (current -> drifts).
  mean      : subtract the per-row time-mean (d6; removes DC=0Hz only).
  highpass  : online causal high-pass (fc << absorber band) on the ANN output = the faithful post-hoc
              BOUNDED-INTEGRAL projection. The first-difference g=Delta(psi) of the construction IS a
              high-pass (zero DC gain, bounded running sum), so a high-pass with cutoff below 130 Hz
              keeps the 130-180 Hz absorber correction but forbids the drift-causing DC/low-freq.
Applied ONLINE (persistent filter state inside the shadow forward) so it is valid inside the closed
rollout, not a replay of a pre-recorded stream.

Two metrics per treatment:
  DRIFT           : tail-mean free-run error on X/Y (the DC problem).
  ABSORBER-BAND   : 130-180 Hz band-pass RMS of (y_model - y_meas) -- absorber capture, independent of
                    drift (the band-pass removes the DC/ramp). Lower = better capture.

Prediction (construction viable): highpass has LOW drift (>= mean-removal) AND preserves the absorber-
band RMS (~ raw), i.e. it kills the drift without harming the 150 Hz correction.

Run:
  CKPT=simulations/gantry_subnet/diagnostics/checkpoints/gantry_drift_last.pth \
  PYTHONIOENCODING=utf-8 <envpython> scripts/gantry/diagnostics-drift/dB_boundedintegral_projection.py
Env: CKPT (required), FC_HZ (30), N_WIN (cap), REC unused (V1).
Outputs -> simulations/gantry_subnet/diagnostics/
"""
import os
import sys
import time

import numpy as np
import torch
from scipy.signal import butter, lfilter, filtfilt
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
from gantry_dynamic.model import build_model
from model_augmentation.fit_systems.blocks import Static_ANN_Block
from gantry_interconnect_dynamic import CFG as cfg

CKPT  = os.environ.get('CKPT', None)
FC_HZ = float(os.environ.get('FC_HZ', '30'))     # high-pass cutoff [Hz], well below the 130-180 band
N_WIN = os.environ.get('N_WIN', None)
if not CKPT:
    raise SystemExit('CKPT required (a drifted X+Theta+Y checkpoint).')

# ── Build pipeline + load drifted checkpoint ─────────────────────────────────
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg); norm = compute_normalization(cfg, data)
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
fit_sys = build_model(cfg.hp, cfg, data, norm)
fit_sys.__dict__ = torch.load(CKPT, map_location='cpu', weights_only=False)
fit_sys.hfn.eval()
ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
route_ix = np.asarray(cfg.ann_route_ix)
ts = cfg.ts_new
absorber_rms = data.val_x_aug[:, 0].std()
print(f'Loaded {os.path.basename(CKPT)}; nw={ann.nw}; fc={FC_HZ} Hz; absorber RMS={absorber_rms:.3e} m')

# ── Online transform machinery on the ANN output (shadow forward) ────────────
_orig_forward = ann.forward
_mode = {'kind': 'raw', 'mean': None, 'zi': None, 'ba': None}


def _shadow_forward(z):
    w = _orig_forward(z)                       # (batch,nw,1)
    wv = w.detach().view(-1, ann.nw).cpu().numpy()
    if _mode['kind'] == 'mean' and _mode['mean'] is not None:
        wv = wv - _mode['mean'][None, :]
    elif _mode['kind'] == 'highpass':
        b, a = _mode['ba']
        out, _mode['zi'] = lfilter(b, a, wv, axis=0, zi=_mode['zi'])  # (1,nw) online, persistent state
        wv = out
    return torch.tensor(wv, dtype=w.dtype).view(w.shape)


ann.forward = _shadow_forward


def _run(kind, mean_vec=None):
    _mode['kind'] = kind
    _mode['mean'] = mean_vec
    if kind == 'highpass':
        b, a = butter(2, FC_HZ / (0.5 / ts), btype='high')
        _mode['ba'] = (b, a)
        _mode['zi'] = np.zeros((max(len(a), len(b)) - 1, ann.nw))
    r = fit_sys.apply_experiment(v1c)
    return np.asarray(r.y, dtype=np.float64)


# ── V1 data ──────────────────────────────────────────────────────────────────
v1 = data.val_data
N  = len(v1.u) if N_WIN is None else min(int(N_WIN), len(v1.u))
v1c = deepSI.System_data(u=np.asarray(v1.u)[:N], y=np.asarray(v1.y)[:N], dt=v1.dt)
y_meas = np.asarray(v1c.y, dtype=np.float64)

# per-row mean from a raw pass (also gives the raw result)
_records = []
_o2 = ann.forward
def _rec_forward(z):
    w = _o2(z); _records.append(w.detach().view(-1, ann.nw).cpu().numpy()); return w
# temporarily wrap to record during the raw run
ann.forward = lambda z: _rec_forward(z)
t0 = time.time(); y_raw = _run('raw'); print(f'  raw run ({time.time()-t0:.0f}s)')
ann.forward = _shadow_forward
mean_w = np.concatenate(_records, axis=0).mean(axis=0)

t0 = time.time(); y_mean = _run('mean', mean_w); print(f'  mean-removed run ({time.time()-t0:.0f}s)')
t0 = time.time(); y_hp   = _run('highpass');      print(f'  high-pass run ({time.time()-t0:.0f}s)')

# ── Metrics ───────────────────────────────────────────────────────────────────
M = min(len(y_raw), len(y_mean), len(y_hp), len(y_meas))
ym = y_meas[-M:]
bb, ab = butter(4, [130 / (0.5/ts), 180 / (0.5/ts)], btype='band')


def metrics(yhat, name):
    e = yhat[-M:] - ym
    tail = np.abs(e[int(0.8*M):].mean(axis=0))               # drift
    eb = filtfilt(bb, ab, e, axis=0)                         # 130-180 Hz band
    band = np.sqrt((eb**2).mean(axis=0))
    print(f'  {name:10s}  drift X1={tail[0]:.2e} X2={tail[1]:.2e} Y={tail[2]:.2e} m   |   '
          f'absorber-band RMS Y={band[2]:.2e} m')
    return tail, band


t_ax = np.arange(M) * ts
print(f'\n=== D-B: drift vs absorber-band capture ({M*ts:.1f}s free-run) ===')
raw_t, raw_b   = metrics(y_raw,  'raw')
mean_t, mean_b = metrics(y_mean, 'mean')
hp_t, hp_b     = metrics(y_hp,   'highpass')
print(f'  reference absorber RMS = {absorber_rms:.2e} m')
print('\n=== Verdict ===')
drift_ok = np.abs(hp_t).max() <= 1.5 * np.abs(mean_t).max()
band_ok  = hp_b[2] <= 1.3 * raw_b[2]
if drift_ok and band_ok:
    print('  -> high-pass (bounded-integral) removes drift AT LEAST as well as mean-removal AND preserves')
    print('     the 130-180 Hz absorber capture -> structural projection compatible with the learned signal.')
else:
    print(f'  -> mixed: drift_ok={drift_ok} (hp Y drift {hp_t[2]:.2e} vs mean {mean_t[2]:.2e}); '
          f'band_ok={band_ok} (hp {hp_b[2]:.2e} vs raw {raw_b[2]:.2e}). Inspect fc / phase.')

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ch, lbl in enumerate(['X1 [m]', 'X2 [m]', 'Y [m]']):
    axes[ch].plot(t_ax, (y_raw[-M:]  - ym)[:, ch], 'C3', lw=0.6, label='raw')
    axes[ch].plot(t_ax, (y_mean[-M:] - ym)[:, ch], 'C1', lw=0.6, label='mean-removed (d6)')
    axes[ch].plot(t_ax, (y_hp[-M:]   - ym)[:, ch], 'C0', lw=0.6, label='high-pass (bounded-integral)')
    axes[ch].axhline(0, color='k', lw=0.5)
    axes[ch].axhline( absorber_rms, color='0.6', ls=':', lw=0.8)
    axes[ch].axhline(-absorber_rms, color='0.6', ls=':', lw=0.8)
    axes[ch].set_ylabel(f'{lbl} error'); axes[ch].grid(True); axes[ch].legend(fontsize=7, loc='upper right')
axes[-1].set_xlabel('Time [s]')
fig.suptitle(f'D-B: raw vs mean-removed vs high-pass (bounded-integral) on the drifted checkpoint '
             f'(fc={FC_HZ} Hz)')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, 'dB_boundedintegral_projection_V1')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', t=t_ax, y_meas=ym, y_raw=y_raw[-M:], y_mean=y_mean[-M:], y_hp=y_hp[-M:],
         raw_t=raw_t, mean_t=mean_t, hp_t=hp_t, raw_b=raw_b, mean_b=mean_b, hp_b=hp_b,
         mean_w=mean_w, fc=FC_HZ, absorber_rms=absorber_rms, ts=ts)
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
