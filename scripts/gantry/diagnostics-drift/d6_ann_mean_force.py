"""
d6_ann_mean_force.py -- the DIRECT test of hypothesis (B) on the actual object:
does the TRAINED ANN carry a nonzero-mean (DC) output on the X/Y rows, and does
removing that mean stop the free-run drift?

d1-d5 cleared the input (A1), the encoder x0 (A2), and the physics, and showed the
K=0 axes turn a nonzero-mean correction into drift. But none of them used the
trained ANN, so "the ANN injects a DC force that ramps" was still a hypothesis.
This measures it.

The ANN output is added additively to the next state xp on the routed rows
(model.py:111), so a nonzero time-mean of the ANN output on a K=0 row (X=0/dX=3,
Y=2/dY=5) is exactly the drift-driving DC bias. We:
  1. Build the pipeline (imports CFG) and optionally load a trained checkpoint.
  2. Free-run V1 (apply_experiment), capturing the ANN output each step by shadowing
     the block's forward.
  3. Report per-routed-row time-mean and RMS (normalized and, for physical rows,
     in state units), flagging the K=0 rows.
  4. Counterfactual: subtract the measured per-row mean from the ANN during a second
     free-run and compare the Y drift. If the drift collapses, the DC component is
     the cause and a mean/DC guardrail is the fix.

NOTE: at epoch 0 the ANN is zero-init (zero output), so a MEANINGFUL result needs a
TRAINED X+Theta+Y checkpoint (the cluster run 69399). Local pre-D-103 checkpoints
are Theta-only and dimension-incompatible; without CKPT this runs the untrained ANN
(mean ~ 0) purely to verify the machinery.

Run:
  CKPT=/path/to/gantry_ckpt_<job>_best.pt \
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d6_ann_mean_force.py
Env: CKPT (checkpoint .pt state_dict; unset = untrained), N_WIN (cap sim length), REC unused (V1).
Outputs -> simulations/gantry_subnet/diagnostics/
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
from gantry_dynamic.model import build_model
from model_augmentation.fit_systems.blocks import Static_ANN_Block
from gantry_interconnect_dynamic import CFG as cfg

CKPT  = os.environ.get('CKPT', None)
N_WIN = os.environ.get('N_WIN', None)

# ── Build pipeline + (optionally) load a trained checkpoint ───────────────────
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
fit_sys = build_model(cfg.hp, cfg, data, norm)

if CKPT:
    # Both the drift-checkpoint maker and the cluster runs save the whole fit_sys
    # __dict__ (checkpoint_save_system: torch.save(self.__dict__, ...)). Restoring
    # __dict__ replaces the freshly-built interconnect with the trained one.
    fit_sys.__dict__ = torch.load(CKPT, map_location='cpu', weights_only=False)
    print(f'Loaded CKPT {os.path.basename(CKPT)} (full __dict__); bestfit={getattr(fit_sys,"bestfit",float("nan")):.6e}')
else:
    print('No CKPT: running UNTRAINED ANN (zero-init) -- machinery check only, expect mean ~ 0.')
fit_sys.hfn.eval()

route_ix = np.asarray(cfg.ann_route_ix)
names8   = np.array(['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a'])
k0_rows  = {0, 2, 3, 5}   # K=0-relevant rows (X/Y position + velocity)
std_x    = norm.std_x.flatten()   # (6,) physical-state normalization

# ── Locate the ANN block and shadow its forward to record / optionally debias ─
ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
_orig_forward = ann.forward
_records = []
_subtract = {'vec': None}   # (nw,) tensor to subtract from the ANN output, or None


def _shadow_forward(z):
    w = _orig_forward(z)
    _records.append(w.detach().view(-1, ann.nw).cpu().numpy())
    if _subtract['vec'] is not None:
        w = w - _subtract['vec'].view(1, ann.nw, 1)
    return w


ann.forward = _shadow_forward


def _freerun(sysdata):
    _records.clear()
    r = fit_sys.apply_experiment(sysdata)
    ann_out = np.concatenate(_records, axis=0) if _records else np.zeros((0, ann.nw))
    return np.asarray(r.y, dtype=np.float64), ann_out


# ── V1 data (optionally truncated) ────────────────────────────────────────────
v1 = data.val_data
N  = len(v1.u) if N_WIN is None else min(int(N_WIN), len(v1.u))
v1c = deepSI.System_data(u=np.asarray(v1.u)[:N], y=np.asarray(v1.y)[:N], dt=v1.dt)
y_meas = np.asarray(v1c.y, dtype=np.float64)
absorber_rms = data.val_x_aug[:, 0].std()

# ── Pass 1: free-run, capture ANN output ─────────────────────────────────────
t0 = time.time()
y_full, ann_out = _freerun(v1c)
print(f'\nFree-run captured {ann_out.shape[0]} ANN calls in {time.time()-t0:.0f}s (nw={ann.nw})')

mean_w = ann_out.mean(axis=0)                       # (nw,) normalized per routed row
rms_w  = np.sqrt((ann_out ** 2).mean(axis=0))
print('\n=== ANN output per routed row (added to next state xp) ===')
print(f"  {'row':10s} {'mean [norm]':>13s} {'rms [norm]':>12s} {'|mean|/rms':>11s} {'mean [phys]':>13s}")
for j, r in enumerate(route_ix):
    phys = f'{mean_w[j]*std_x[r]:.3e}' if r < 6 else '   (aug)'
    flag = '  <-- K=0' if r in k0_rows else ''
    ratio = abs(mean_w[j]) / (rms_w[j] + 1e-30)
    print(f'  {names8[r]:10s} {mean_w[j]:>13.3e} {rms_w[j]:>12.3e} {ratio:>11.3f} {phys:>13s}{flag}')

k0_j = [j for j, r in enumerate(route_ix) if r in k0_rows]
if k0_j:
    worst = max(k0_j, key=lambda j: abs(mean_w[j]))
    print(f'\n  Largest K=0-row mean: {names8[route_ix[worst]]} = {mean_w[worst]:.3e} [norm]  '
          f'(|mean|/rms = {abs(mean_w[worst])/(rms_w[worst]+1e-30):.2f})')

# ── Pass 2 (counterfactual): subtract the per-row mean, re-run ───────────────
_subtract['vec'] = torch.tensor(mean_w, dtype=next(ann.parameters()).dtype)
y_debias, _ = _freerun(v1c)
_subtract['vec'] = None

# align lengths (apply_experiment drops the encoder warm-up window)
M = min(len(y_full), len(y_debias), len(y_meas))
yf, yd, ym = y_full[-M:], y_debias[-M:], y_meas[-M:]
tail = slice(int(0.8 * M), M)
tf = np.abs((yf - ym)[tail].mean(axis=0))
td = np.abs((yd - ym)[tail].mean(axis=0))
print('\n=== Free-run Y-drift: full ANN vs ANN with per-row mean removed ===')
print(f"  {'chan':4s} {'full |tail|':>13s} {'debiased |tail|':>16s}")
for c, lbl in enumerate(['X1', 'X2', 'Y']):
    print(f'  {lbl:4s} {tf[c]:>13.3e} {td[c]:>16.3e}')
print(f'  absorber RMS = {absorber_rms:.3e} m')
if not CKPT:
    print('  (untrained ANN: both ~ baseline; run with a trained CKPT for the real verdict.)')
elif tf.max() > 3 * absorber_rms and td.max() < 0.5 * tf.max():
    print('  -> removing the ANN mean collapses the drift: the DC component IS the cause (B confirmed).')
    print('     Fix = mean/DC guardrail on the ANN X/Y rows (keeps X/Y routing, D-103).')
else:
    print('  -> removing the mean does NOT collapse the drift: the drift is not a pure DC bias; look')
    print('     at the oscillatory/structural ANN contribution and the nonlinear M(Y) coupling.')

# ── Plot ─────────────────────────────────────────────────────────────────────
t = np.arange(M) * cfg.ts_new
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ch, (ax, lab) in enumerate(zip(axes, ['X1 [m]', 'X2 [m]', 'Y [m]'])):
    ax.plot(t, (yf - ym)[-M:][:, ch], 'C3', lw=0.7, label='full ANN')
    ax.plot(t, (yd - ym)[-M:][:, ch], 'C0', lw=0.7, label='ANN mean removed')
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline( absorber_rms, color='0.5', ls=':', lw=0.8)
    ax.axhline(-absorber_rms, color='0.5', ls=':', lw=0.8,
               label='+/- absorber RMS' if ch == 0 else None)
    ax.set_ylabel(f'{lab} error'); ax.grid(True); ax.legend(fontsize=7, loc='upper right')
axes[-1].set_xlabel('Time [s]')
tag = os.path.basename(CKPT).split('.')[0] if CKPT else 'untrained'
fig.suptitle(f'V1: does removing the ANN per-row mean stop the drift?  ({tag})')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, f'd6_ann_mean_force_{tag}')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', route_ix=route_ix, mean_w=mean_w, rms_w=rms_w, ann_out=ann_out,
         tf=tf, td=td, absorber_rms=absorber_rms, ts=cfg.ts_new, ckpt=str(CKPT))
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
