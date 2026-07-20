"""
d14_datasilence_estimator_alignment.py (v2) -- Layer-2 PRE-BUILD diagnostic:
at OPERATIONAL magnitudes, is the model's DC component loss-free to pin, while
its absorber-band component is loss-expensive to touch?

v1 DESIGN FLAW (kept for the record): ranking directions by raw sensitivity
|dRMS|/eps across probe FREQUENCIES conflates data-informativeness with PLANT
GAIN -- a DC increment on a K=0 velocity row integrates into a position ramp
(large output effect) while a 150 Hz increment integrates into a tiny bounded
oscillation (~1/omega attenuation). The v1 "separation" verdict was therefore
gain-confounded and meaningless.

v2 METRIC: probe each direction AT THE SIZE IT EXISTS IN THE TRAINED MODEL.
  * DC components: REMOVE the measured per-row mean (exactly what Layer 2
    would do to the drift direction).
  * Absorber-band components: INJECT a 150 Hz probe at the row's MEASURED
    band amplitude (surrogate for corrupting the learned band content).
    # HEURISTIC: random-phase same-amplitude injection as the corruption
    # surrogate -- we cannot phase-cancel the model's own band content with a
    # fixed per-step probe; a same-size disturbance measures the loss's
    # sensitivity to that component at operational scale. Both phases probed.
  S_op(v) = paired per-window Delta RMS / ref RMS   [dimensionless, with SE]

PRE-DECLARED reading (the Layer-2 safety question, limits doc B1/C1):
  S_op(DC removals) ~ 0 within ~2 SE  AND  S_op(band probes) >> 2 SE positive
    -> pinning the DC is loss-free, touching the band is loss-expensive: the
       penalty can separate them at operational scale; Layer 2 is safe to build.
  S_op(DC removal) significantly positive -> pinning fights the data (B1 risk).
  S_op(band probes) ~ 0 -> the loss does not defend the band; the "penalty
       vanishes on informed directions" argument loses its empirical support
       at this operating point (C1-adjacent risk); rethink before building.

Probes (route rows; nw = len(cfg.ann_route_ix)):
  A  remove FULL mean_w vector      (the d12 measurement, now at higher power)
  B  remove DC on dY row only       (the drift driver row, d6/d12)
  C+/C- inject 150 Hz on dY row     at measured dY-row band amplitude
  D+/D- inject 150 Hz on dTheta row at measured dTheta-row band amplitude
  E+/E- inject 150 Hz on delta_a row at measured delta_a-row band amplitude
                                    (the absorber's home row)

Run:
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d14_datasilence_estimator_alignment.py
Env: CKPT (default gantry_drift_last.pth), TRAJS (default "T3,T10"),
     N_WINDOWS (per traj, default 20), NF (default 400).
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
from scipy import signal as sps

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
FA        = dc.fa
BAND      = (120.0, 180.0)

# ── Pipeline + trained checkpoint ─────────────────────────────────────────────
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

# ── ANN shadow: w + amp * v * s(t), s = 1 (DC) or sin(2 pi f (k ts) + phase) ──
ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
_orig_forward = ann.forward
_probe = {'vec': None, 'freq': 0.0, 'amp': 0.0, 'phase': 0.0, 'k': 0}
_records = []


def _shadow_forward(z):
    w = _orig_forward(z)
    _records.append(w.detach().view(-1, ann.nw).cpu().numpy())
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


def _windowed_rms(vec=None, freq=0.0, amp=0.0, phase=0.0, capture=False):
    """Per-window pooled RMS over the training windows (paired across probes)."""
    _probe.update(vec=vec, freq=freq, amp=amp, phase=phase)
    if capture:
        _records.clear()
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
    out = np.asarray(pooled)
    if capture:
        ann_out = np.concatenate(_records, axis=0) if _records else np.zeros((0, ann.nw))
        return out, ann_out
    return out


# ── Reference pass: fit baseline + operational magnitudes ─────────────────────
print(f'\nReference pass ({TRAJ_TAGS} x {N_WINDOWS} windows @ nf={NF}) ...')
t0 = time.time()
rms_ref, ann_out = _windowed_rms(capture=True)
ref = rms_ref.mean()
print(f'  {len(rms_ref)} windows, ref pooled RMS = {ref:.6e}  ({time.time()-t0:.0f}s)')

mean_w = ann_out.mean(axis=0)                              # per-row DC [norm]
sos = sps.butter(4, BAND, btype='bandpass', fs=1.0 / ts, output='sos')
band_rms = np.sqrt((sps.sosfiltfilt(sos, ann_out, axis=0) ** 2).mean(axis=0))
print('\n  operational magnitudes per routed row [norm]:')
print(f"  {'row':10s} {'DC (mean_w)':>13s} {'150Hz-band RMS':>15s}")
for jj, r in enumerate(route_ix):
    print(f'  {names8[r]:10s} {mean_w[jj]:>13.3e} {band_rms[jj]:>15.3e}')

# ── Probes at operational magnitude ───────────────────────────────────────────
def _unit(ix):
    v = np.zeros(nw); v[j[ix]] = 1.0
    return v

dt_ = next(ann.parameters()).dtype
mw_t = torch.tensor(mean_w, dtype=dt_)
amp_band = {r: band_rms[j[r]] * np.sqrt(2) for r in (5, 4, 6)}   # sine amp from RMS
PROBES = [
    ('A  remove full mean_w',        dict(vec=mw_t, freq=0.0, amp=-1.0)),
    ('B  remove DC dY row',          dict(vec=torch.tensor(_unit(5), dtype=dt_),
                                          freq=0.0, amp=-float(mean_w[j[5]]))),
    ('C+ 150Hz dY @op-amp',          dict(vec=torch.tensor(_unit(5), dtype=dt_),
                                          freq=FA, amp=amp_band[5], phase=0.0)),
    ('C- 150Hz dY @op-amp (pi)',     dict(vec=torch.tensor(_unit(5), dtype=dt_),
                                          freq=FA, amp=amp_band[5], phase=np.pi)),
    ('D+ 150Hz dTheta @op-amp',      dict(vec=torch.tensor(_unit(4), dtype=dt_),
                                          freq=FA, amp=amp_band[4], phase=0.0)),
    ('D- 150Hz dTheta @op-amp (pi)', dict(vec=torch.tensor(_unit(4), dtype=dt_),
                                          freq=FA, amp=amp_band[4], phase=np.pi)),
    ('E+ 150Hz delta_a @op-amp',     dict(vec=torch.tensor(_unit(6), dtype=dt_),
                                          freq=FA, amp=amp_band[6], phase=0.0)),
    ('E- 150Hz delta_a @op-amp (pi)', dict(vec=torch.tensor(_unit(6), dtype=dt_),
                                           freq=FA, amp=amp_band[6], phase=np.pi)),
]

print(f"\n{'probe':32s} {'S_op = dRMS/refRMS':>19s} {'dRMS':>11s} {'SE(paired)':>11s} {'d/SE':>7s}")
res = {}
for name, kw in PROBES:
    t0 = time.time()
    r = _windowed_rms(**kw)
    d = r - rms_ref                                    # paired per-window
    se = d.std(ddof=1) / np.sqrt(len(d))
    res[name] = (d.mean() / ref, d.mean(), se, d.mean() / se if se > 0 else np.nan)
    print(f'{name:32s} {res[name][0]:>19.3e} {d.mean():>11.3e} {se:>11.2e} '
          f'{res[name][3]:>7.2f}   ({time.time()-t0:.0f}s)')

# ── Verdict (pre-declared) ────────────────────────────────────────────────────
dc_sig   = max(abs(res['A  remove full mean_w'][3]), abs(res['B  remove DC dY row'][3]))
band_sig = min(res[k][3] for k in res if k.startswith(('C', 'D', 'E')))
print(f'\nDC-removal worst |d/SE| = {dc_sig:.2f}   band-probe min d/SE = {band_sig:.2f}')
if dc_sig < 2 and band_sig > 2:
    print('-> SEPARATED at operational scale: pinning the DC is loss-free; touching the')
    print('   absorber band is loss-expensive. Layer 2 is safe to build on this basis.')
elif dc_sig >= 2:
    print('-> DC removal costs fit (B1 risk): the drift component is NOT loss-free at')
    print('   operational scale on these windows; reconcile with d12 (n=120) before building.')
else:
    print('-> Band probes not defended by the loss at operational scale: the "penalty')
    print('   vanishes on informed directions" support is weak here; investigate before building.')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
labels = list(res)
vals   = [res[k][0] for k in labels]
errs   = [2 * res[k][2] / ref for k in labels]
colors = ['C0' if k.startswith(('A', 'B')) else 'C3' for k in labels]
ax.bar(np.arange(len(labels)), vals, yerr=errs, capsize=3, color=colors)
ax.axhline(0, color='k', lw=0.6)
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
ax.set_ylabel('S_op = paired dRMS / ref RMS  (2 SE bars)')
ax.set_title('d14 v2: fit impact of pinning DC (blue) vs corrupting the 150 Hz band (red), '
             'each at its operational magnitude')
ax.grid(True, axis='y')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, 'd14_datasilence_estimator_alignment')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', labels=np.array(labels),
         S_op=np.array([res[k][0] for k in labels]),
         dmean=np.array([res[k][1] for k in labels]),
         se=np.array([res[k][2] for k in labels]),
         dse=np.array([res[k][3] for k in labels]),
         mean_w=mean_w, band_rms=band_rms, route_ix=route_ix, ref_rms=ref,
         rms_ref_windows=rms_ref, nf=NF, n_windows=N_WINDOWS,
         trajs=np.array(TRAJ_TAGS), ckpt=str(CKPT))
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
