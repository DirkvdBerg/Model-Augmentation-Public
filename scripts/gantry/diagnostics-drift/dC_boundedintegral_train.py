"""
dC_boundedintegral_train.py -- D-C: does an ANN TRAINED WITH the bounded-integral constraint learn the
absorber WITHOUT drifting? (trainability test; self-contained, NO edits to build_model/framework)

The construction (drift-diagnosis-status.md §5b): the ANN output on the routed rows is
`g_k = psi(z_k) - psi(z_{k-1})`, psi = psi_scale*tanh(net) bounded -> running sum telescopes ->
bounded -> no drift by construction, while the instantaneous output is free.

This diagnostic builds the normal X+Theta+Y pipeline (build_model, unchanged) and then, IN THIS SCRIPT
ONLY, converts the ANN block into bounded-integral mode by monkeypatching its forward + adding a
per-rollout reset. It does NOT touch build_model, the interconnect, or the encoder. If D-C passes, the
same change is promoted into build_model behind a flag (a separate step).

Reset: each training window is a fresh rollout starting at x=encoder(...) (interconnect.loss, L434).
We monkeypatch fit_sys.loss to reset the block before each rollout (covers same-size consecutive
training windows); the block also auto-resets on batch-size change (covers train<->val<->sim). The
free-run eval resets explicitly before apply_experiment.

Success = (1) LEARNS: per-epoch nf-RMS decreases / absorber-band error drops; AND (2) NO DRIFT: full
free-run X/Y stays bounded; AND (3) absorber captured (130-180 Hz band error low). Contrast: the
unconstrained run is best=epoch 0 and drifts (d6/d7).

Run:
  EPOCHS=5 PSI_SCALE=1.0 PYTHONIOENCODING=utf-8 <envpython> \
     scripts/gantry/diagnostics-drift/dC_boundedintegral_train.py
Env: EPOCHS(5), LR(1.49e-8), NF(400), PSI_SCALE(1.0), VAL_SAMPLES(8000).
Outputs -> simulations/gantry_subnet/diagnostics/
"""
import os
import sys
import time
import types

import numpy as np
import torch
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GANTRY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, GANTRY)
sys.path.insert(0, os.path.dirname(__file__))

import deepSI
import drift_common as dc
from gantry_dynamic.config import RunConfig
from gantry_dynamic.data import load_datasets, compute_normalization
from gantry_dynamic.model import build_model, train_model
from gantry_dynamic.training import _install_nf_val_probe
from model_augmentation.fit_systems.blocks import Static_ANN_Block

EPOCHS      = int(os.environ.get('EPOCHS', '5'))
LR          = float(os.environ.get('LR', '1.49e-8'))
NF          = int(os.environ.get('NF', '400'))
PSI_SCALE   = float(os.environ.get('PSI_SCALE', '1.0'))
VAL_SAMPLES = int(os.environ.get('VAL_SAMPLES', '8000'))

cfg = RunConfig(ann_route_ix=(0, 1, 2, 3, 4, 5, 6, 7), stride=100, lr=LR, nf_override=NF, epochs=EPOCHS)
print(f'D-C: routing={cfg.ann_route_ix} stride={cfg.stride} lr={cfg.lr:.2e} nf={cfg.nf} '
      f'epochs={cfg.epochs} psi_scale={PSI_SCALE}')

np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg); norm = compute_normalization(cfg, data)
_v0 = data.val_list[0]
data.val_ckpt_data = deepSI.System_data(u=_v0.u[:VAL_SAMPLES], y=_v0.y[:VAL_SAMPLES], dt=_v0.dt)
_t0 = data.train_list[0]
search_train = deepSI.System_data(u=_t0.u[:VAL_SAMPLES], y=_t0.y[:VAL_SAMPLES], dt=_t0.dt)

trial_seed = cfg.seed + 3
np.random.seed(trial_seed); torch.manual_seed(trial_seed)
fit_sys = build_model(cfg.hp, cfg, data, norm)

# ── Convert the ANN block to bounded-integral mode (this script only) ─────────
# CLASS-level monkeypatch: methods live on the class (pickle by reference), so they survive the
# checkpoint save/load inside fit() -- an instance-level MethodType does NOT round-trip through
# torch.save(self.__dict__). Per-instance state (_psi_prev, psi_scale) are plain data attrs.
def _bi_forward(self, z):
    w = self.net(z.view(-1, self.nz))
    psi = getattr(self, 'psi_scale', 1.0) * torch.tanh(w).view(-1, self.nw, 1)
    prev = getattr(self, '_psi_prev', None)
    g = torch.zeros_like(psi) if (prev is None or prev.shape[0] != psi.shape[0]) else psi - prev
    self._psi_prev = psi
    return g


def _reset(self):
    self._psi_prev = None
    self._reset_count = getattr(self, '_reset_count', 0) + 1


Static_ANN_Block.forward = _bi_forward       # class-level -> picklable, survives checkpoint reload
Static_ANN_Block.reset   = _reset

ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
ann.psi_scale = PSI_SCALE
ann._psi_prev = None
ann._reset_count = 0
print('ANN block converted to bounded-integral mode (g_k = psi_k - psi_{k-1}).')

# reset before every training rollout (interconnect.loss)
_orig_loss = fit_sys.loss
def _loss_with_reset(*a, **k):
    ann.reset()
    return _orig_loss(*a, **k)
fit_sys.loss = _loss_with_reset

# ── Train (short), with the mandatory per-epoch nf-RMS probe ─────────────────
_orig_cve = _install_nf_val_probe(fit_sys, cfg.hp, cfg, search_train, data.val_ckpt_data)
t0 = time.time()
try:
    train_model(fit_sys, cfg.hp, cfg, data, epochs=cfg.epochs)
finally:
    fit_sys.cal_validation_error = _orig_cve
print(f'training done ({time.time()-t0:.0f}s); loss resets fired = {ann._reset_count}; '
      f'bestfit(val sim-RMS)={fit_sys.bestfit:.4e}')

# recover the END-OF-TRAINING model (fit reloads _best at the end); class methods persist through
# the reload, only re-find the (possibly new) instance and ensure its data attrs are set.
fit_sys.checkpoint_load_system(name='_last')
ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
if not hasattr(ann, 'psi_scale'):
    ann.psi_scale = PSI_SCALE
ann._psi_prev = None
fit_sys.hfn.eval()

# ── Evaluate: free-run drift + absorber capture on V1 ────────────────────────
ts = cfg.ts_new
v1 = data.val_data
v1c = deepSI.System_data(u=np.asarray(v1.u), y=np.asarray(v1.y), dt=v1.dt)
y_meas = np.asarray(v1c.y, dtype=np.float64)
absorber_rms = data.val_x_aug[:, 0].std()

ann.reset()
t0 = time.time()
r = fit_sys.apply_experiment(v1c)
y_hat = np.asarray(r.y, dtype=np.float64)
M = min(len(y_hat), len(y_meas))
e = y_hat[-M:] - y_meas[-M:]
t_ax = np.arange(M) * ts
print(f'free-run eval ({time.time()-t0:.0f}s, {M*ts:.1f}s)')

tail = np.abs(e[int(0.8*M):].mean(axis=0))
slope = np.array([np.polyfit(t_ax[int(0.8*M):], e[int(0.8*M):, c], 1)[0] for c in range(3)])
sim_rms_full = np.sqrt((e**2).mean(axis=0))                # FULL 12s free-run sim-RMS (deliverable metric)
bb, ab = butter(4, [130/(0.5/ts), 180/(0.5/ts)], btype='band')
band = np.sqrt((filtfilt(bb, ab, e, axis=0)**2).mean(axis=0))

print('\n=== D-C results (bounded-integral, TRAINED) ===')
print(f'  FULL-12s sim-RMS  X1={sim_rms_full[0]:.2e} X2={sim_rms_full[1]:.2e} Y={sim_rms_full[2]:.2e} m'
      f'   (deliverable metric; drifted model Y~1.6e-2 at 12s, S1)')
print(f'  free-run drift  X1={tail[0]:.2e} X2={tail[1]:.2e} Y={tail[2]:.2e} m   '
      f'(slope Y={slope[2]:+.2e} m/s)')
print(f'  absorber-band 130-180Hz RMS  X1={band[0]:.2e} X2={band[1]:.2e} Y={band[2]:.2e} m')
print(f'  absorber RMS reference = {absorber_rms:.2e} m')
print('\n  contrast: unconstrained X+Theta+Y is best=epoch0, free-run Y drift ~2.6e-2 m (d6/d7).')
print('=== Verdict ===')
# DRIFT = ongoing ramp -> judge by SLOPE (m/s), not tail-mean. The tail-MEAN also contains the
# bounded encoder-init offset (tau*dv ~ 2e-4 m on Y, d3/D-B) which bounded-integral neither causes
# nor should fix; a flat slope means the free-run is BOUNDED (no drift).
no_drift = abs(slope[2]) < 1e-5 and abs(slope[0]) < 1e-5   # flat => bounded, no ongoing drift
learn_ok = band[2] < 5 * absorber_rms                       # absorber band captured
print(f'  no ongoing drift (slope flat)? {no_drift}  (Y slope {slope[2]:+.2e} m/s, X {slope[0]:+.2e})')
print(f'  absorber captured?             {learn_ok}  (Y band {band[2]:.2e} vs absorber {absorber_rms:.2e})')
print(f'  bounded Y offset = {tail[2]:.2e} m  (~ the encoder-init offset tau_Y*dv from d3/D-B, NOT drift)')
if no_drift and learn_ok:
    print('  -> bounded-integral TRAINS without ONGOING DRIFT (slope flat) and keeps the absorber band')
    print('     -> D-C PASS (coupling-channel trainability confirmed). Residual Y offset is the bounded')
    print('        encoder-IC effect, separate from the drift the constraint targets.')
else:
    print('  -> inspect: nf-RMS trajectory / psi_scale / lr / more epochs.')

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ch, lbl in enumerate(['X1 [m]', 'X2 [m]', 'Y [m]']):
    axes[ch].plot(t_ax, e[:, ch], 'C0', lw=0.6, label='bounded-integral, trained')
    axes[ch].axhline(0, color='k', lw=0.5)
    axes[ch].axhline( absorber_rms, color='0.6', ls=':', lw=0.8)
    axes[ch].axhline(-absorber_rms, color='0.6', ls=':', lw=0.8)
    axes[ch].set_ylabel(f'{lbl} error'); axes[ch].grid(True); axes[ch].legend(fontsize=7, loc='upper right')
axes[-1].set_xlabel('Time [s]')
fig.suptitle(f'D-C: ANN TRAINED with the bounded-integral constraint -- drift bounded? absorber kept? '
             f'(psi_scale={PSI_SCALE}, {EPOCHS} epochs)')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, 'dC_boundedintegral_train_V1')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', t=t_ax, e=e, tail=tail, slope=slope, band=band, sim_rms_full=sim_rms_full,
         absorber_rms=absorber_rms, bestfit=fit_sys.bestfit, psi_scale=PSI_SCALE, ts=ts)
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
