"""
demo7_g9_intervention.py -- G9 (plan doc §14.3): the interventional close of the causal chain.

Title question: "If we enforce the zero-mean force Jan expects, does the drift ever form?"
Runs AFTER the MODE='zeromean' cluster job returns its checkpoint.

  (a) 12 s V1 free-run Y error: red "trained, unconstrained" (control = 71013 trial 3 or
      70903 rung 0) vs blue "trained with the zero-mean pin" vs grey "no ANN".
  (b) window-fit trajectories of pinned vs control (expect overlap within ~2%).
Prints: per-row trained means (pinned vs control), free-run envelope ratios.

Env: PIN_CKPT (path to zeromean_nf400_last.pth; default = newest zeromean_*/ under
augmentation_linear_map), CTRL_CKPT (default 71013 trial3 nf=800... NOTE the honest control
at the SAME nf=400 is 70903 rung 0, warm; trial3 is fresh-init nf=800 -- both printed).
Run: conda run -n GraduationProject python scripts/gantry/drift-demo/demo7_g9_intervention.py
"""
__project_origin__ = "added"

import glob
import os
import sys

import numpy as np
import torch
import deepSI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo_common as dm
from demo_common import CFG, REPO
from model_augmentation.fit_systems.blocks import Static_ANN_Block

ALM = os.path.join(REPO, 'simulations', 'gantry_subnet', 'augmentation_linear_map')
_pins = sorted(glob.glob(os.path.join(ALM, 'zeromean_*', 'zeromean_nf400_last.pth')))
PIN_CKPT = os.environ.get('PIN_CKPT', _pins[-1] if _pins else '')
CTRL_CKPT = os.environ.get('CTRL_CKPT',
                           os.path.join(ALM, 'curriculum_70903', 'rung0_nf400_last.pth'))
if not os.path.exists(PIN_CKPT):
    sys.exit('pinned checkpoint not found -- run the MODE=zeromean cluster job first '
             '(or set PIN_CKPT)')

fit_sys, norm, K0, na, nb, na_right, nb_right = dm.build_pipeline()
u_v, y_v, _, da_v = dm.load_T('V1_standstill_Yp10.mat', need_absorber=True)
absorber = float(da_v.std())
ts = CFG.ts_new
N = len(u_v)


def freerun(ckpt, zero_ann=False):
    fit_sys.__dict__ = torch.load(ckpt, map_location='cpu', weights_only=False)
    fit_sys.hfn.eval()
    ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    rec = []
    orig = ann.forward

    def shadow(z):
        w = orig(z)
        rec.append(w.detach().view(-1, ann.nw).cpu().numpy())
        return w * 0.0 if zero_ann else w

    ann.forward = shadow
    try:
        r = fit_sys.apply_experiment(deepSI.System_data(u=u_v, y=y_v, dt=ts))
    finally:
        ann.forward = orig
    hist = {k: np.asarray(fit_sys.__dict__.get(k, []), dtype=float)
            for k in ('Loss_train_nf', 'Loss_val_nf', 'Loss_val')}
    return np.asarray(r.y, dtype=np.float64), np.concatenate(rec, axis=0), hist


print('free-running pinned / control / no-ANN (3 x 12 s) ...')
y_pin, w_pin, h_pin = freerun(PIN_CKPT)
y_ctl, w_ctl, h_ctl = freerun(CTRL_CKPT)
y_off, _, _ = freerun(CTRL_CKPT, zero_ann=True)
M = min(map(len, (y_pin, y_ctl, y_off)))
ym = y_v[N - M:N]
e_pin, e_ctl, e_off = (y[-M:] - ym for y in (y_pin, y_ctl, y_off))
t = np.arange(M) * ts

env = lambda e: np.sqrt((e[2*M//3:, 2]**2).mean()) / max(np.sqrt((e[M//3:2*M//3, 2]**2).mean()), 1e-30)
print('\n=== dY-row output mean / |mean|/rms ===')
for lbl, w in (('pinned', w_pin), ('control', w_ctl)):
    m, r = w[:, 5].mean(), np.sqrt((w[:, 5]**2).mean())
    print(f'  {lbl:8s} mean={m:+.3e}  |mean|/rms={abs(m)/(r+1e-30):.3f}')
print('=== Y tail |mean| [m] / envelope ratio (>1.2 = drifting) ===')
for lbl, e in (('pinned', e_pin), ('control', e_ctl), ('no ANN', e_off)):
    print(f'  {lbl:8s} tail={abs(e[int(0.8*M):, 2].mean()):.3e}  envelope={env(e):.2f}')

fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5))
a1.plot(t, e_ctl[:, 2] * 100, color='#c62828', lw=1.0)
a1.plot(t, e_pin[:, 2] * 100, color='#1565c0', lw=1.0, ls='--')
a1.plot(t, e_off[:, 2] * 100, color='0.45', lw=1.0, ls=':')
for lbl, e, col, yfr in (('trained, unconstrained', e_ctl, '#c62828', 0.25),
                         ('trained WITH zero-mean pin', e_pin, '#1565c0', 0.72),
                         ('no ANN', e_off, '0.45', 0.85)):
    a1.annotate(lbl, xy=(0.55, yfr), xycoords='axes fraction', fontsize=9,
                color=col, fontweight='bold')
a1.set_ylabel('Y error [cm]'); a1.set_xlabel('time [s]'); a1.grid(True)
a1.set_title('free-run: does the drift ever form when the mean is pinned during training?')
for h, col, lbl in ((h_ctl, '#c62828', 'unconstrained'), (h_pin, '#1565c0', 'with pin')):
    if h['Loss_train_nf'].size:
        a2.plot(h['Loss_train_nf'] * 1e3, color=col, marker='o', ms=3, lw=1.2, label=lbl)
a2.set_ylabel('train window fit [mm]'); a2.set_xlabel('epoch'); a2.grid(True)
a2.legend(fontsize=8)
a2.set_title('the fit cost of the pin (prediction: within ~2%, the mean was loss-neutral)')
fig.suptitle('If we enforce the zero-mean force Jan expects, does the drift ever form?')
dm.add_provenance(fig, f'pinned: {os.path.basename(PIN_CKPT)} | control: '
                       f'{os.path.basename(CTRL_CKPT)} | V1, 12 s free-run')
fig.tight_layout()
p = os.path.join(dm.OUT_DIR, 'g9_intervention.png')
fig.savefig(p, dpi=150)
np.savez(os.path.join(dm.OUT_DIR, 'g9_intervention.npz'), t=t, e_pin=e_pin, e_ctl=e_ctl,
         e_off=e_off, w_pin_dY=w_pin[:, 5], w_ctl_dY=w_ctl[:, 5])
print(f'Saved: {p}')
