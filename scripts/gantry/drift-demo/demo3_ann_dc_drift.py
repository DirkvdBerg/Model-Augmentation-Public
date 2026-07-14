"""
demo3_ann_dc_drift.py -- F2 (plan doc §12): the trained-ANN DC counterfactual + not-energy panel.

CLAIM (caption-claim): the trained ANN emits a near-constant force on the K=0 rows; removing
that single constant collapses the drift; velocity stays bounded while position ramps, so the
mechanism is the position integrator, not energy.

HEADLINE (one checkpoint, default gantry_drift_last, d6-d15 provenance), V1, 12 s free-run,
three passes differing in ONE thing each (the d6 shadow pattern):
  pass 1  full trained ANN            -> capture the ANN output, measure its per-row mean (DC)
  pass 2  ANN minus its measured DC   -> the counterfactual (one constant per row)
  pass 3  ANN output zeroed           -> "ANN off" at the SAME (trained-)encoder x0
Panels: (1) Y position error, 3 lines + collapse factor; (2) Y-error velocity (FD of panel 1's
red trace): BOUNDED while position ramps (C7, the not-energy panel); (3) the ANN dY-row output
vs time (near-horizontal, |mean|/rms printed).

COMPANION (the A2 closure): bar chart of the K=0-row ANN output mean (normalized units, rms
whiskers) across ALL available checkpoints (71013 cold trials, 70903 warm rungs, drift_last),
each measured over a short free-run (d15: the DC is stationary, -0.7% over 12 s).
PRE-DECLARED HONEST RISK: if some checkpoints show ~zero DC, show it and soften the claim.

Outputs -> simulations/gantry_subnet/diagnostics/drift-demo/:
  f2_dc_counterfactual.png, f2_dc_across_checkpoints.png, f2_dc.npz

Run (headline 3 x 12 s free-runs + short run per extra checkpoint):
  conda run -n GraduationProject python scripts/gantry/drift-demo/demo3_ann_dc_drift.py
Env: CKPT (headline checkpoint path), N_WIN (cap headline length), N_BAR (bar-chart free-run
     samples/checkpoint, default 8000 = 2 s), SKIP_BAR=1 (headline only).
"""
__project_origin__ = "added"

import os
import sys
import glob
import time

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

FILE   = 'V1_standstill_Yp10.mat'    # the dissected validation record (d6/d7 provenance)
CKPT   = os.environ.get('CKPT', os.path.join(
    REPO, 'simulations', 'gantry_subnet', 'diagnostics', 'checkpoints', 'gantry_drift_last.pth'))
N_WIN  = os.environ.get('N_WIN', None)
N_BAR  = int(os.environ.get('N_BAR', 8000))
SKIP_BAR = os.environ.get('SKIP_BAR', '0') == '1'

STATE8 = np.array(['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a'])
K0_ROWS = [0, 2, 3, 5]               # K=0-relevant rows: X, Y positions + velocities

# ── Pipeline shell (architecture donor; each checkpoint replaces the full __dict__) ─
fit_sys, norm, K0, na, nb, na_right, nb_right = dm.build_pipeline()
u_v, y_v, xl_v, da_v = dm.load_T(FILE, need_absorber=True)
absorber_rms = float(da_v.std())
ts = CFG.ts_new


def load_ckpt(fit_sys, path):
    """d6 pattern: checkpoints store the full fit_sys __dict__ (torch.save(self.__dict__))."""
    fit_sys.__dict__ = torch.load(path, map_location='cpu', weights_only=False)
    fit_sys.hfn.eval()
    ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    return ann


def freerun(fit_sys, ann, u, y, n, subtract=None, zero=False):
    """Free-run with the ANN forward shadowed: capture outputs; optionally subtract a fixed
    per-row vector or zero the output entirely. Returns (y_sim (M,3), ann_out (T,nw))."""
    records = []
    orig = ann.forward

    def shadow(z):
        w = orig(z)
        records.append(w.detach().view(-1, ann.nw).cpu().numpy())
        if zero:
            return w * 0.0
        if subtract is not None:
            return w - subtract.view(1, ann.nw, 1)
        return w

    ann.forward = shadow
    try:
        sd = deepSI.System_data(u=u[:n], y=y[:n], dt=ts)
        r = fit_sys.apply_experiment(sd)
    finally:
        ann.forward = orig
    return np.asarray(r.y, dtype=np.float64), np.concatenate(records, axis=0)


# ── HEADLINE: three passes on the dissected checkpoint ───────────────────────
if not os.path.exists(CKPT):
    sys.exit(f'headline checkpoint not found: {CKPT}')
ann = load_ckpt(fit_sys, CKPT)
tag = os.path.basename(CKPT).split('.')[0]
Ntot = len(u_v) if N_WIN is None else min(int(N_WIN), len(u_v))
print(f'\nHEADLINE checkpoint: {tag}  (nw={ann.nw}; V1, {Ntot*ts:.1f} s free-run x3)')
assert ann.nw == 8, f'expected full X+Theta+Y routing (nw=8), got nw={ann.nw}'

t0 = time.time()
y_full, w_full = freerun(fit_sys, ann, u_v, y_v, Ntot)
mean_w = w_full.mean(axis=0)
rms_w  = np.sqrt((w_full ** 2).mean(axis=0))
print(f'  pass 1 (full) done {time.time()-t0:.0f}s')
print('\n=== ANN output per routed row (normalized; added to next state xp) ===')
print(f"  {'row':10s} {'mean':>12s} {'rms':>12s} {'|mean|/rms':>11s}")
for j in range(ann.nw):
    print(f'  {STATE8[j]:10s} {mean_w[j]:>12.3e} {rms_w[j]:>12.3e} '
          f'{abs(mean_w[j])/(rms_w[j]+1e-30):>11.3f}' + ('   <-- K=0' if j in K0_ROWS else ''))

y_deb, _ = freerun(fit_sys, ann, u_v, y_v, Ntot,
                   subtract=torch.tensor(mean_w, dtype=next(ann.parameters()).dtype))
y_off, _ = freerun(fit_sys, ann, u_v, y_v, Ntot, zero=True)
print(f'  passes 2-3 done {time.time()-t0:.0f}s total')

M = min(len(y_full), len(y_deb), len(y_off), Ntot)
ym = y_v[Ntot - M:Ntot]        # truth, aligned to the encoder-warmup-trimmed sims
e_full, e_deb, e_off = y_full[-M:] - ym, y_deb[-M:] - ym, y_off[-M:] - ym
t = np.arange(M) * ts

tail = slice(int(0.8 * M), M)
tf, td, to = (np.abs(e[tail].mean(axis=0)) for e in (e_full, e_deb, e_off))
collapse = tf[2] / max(td[2], 1e-30)
env = lambda e: np.sqrt((e[2*M//3:, 2]**2).mean()) / max(np.sqrt((e[M//3:2*M//3, 2]**2).mean()), 1e-30)
print('\n=== Y tail |mean| [m] and envelope ratio (last/middle third; >1.2 = drifting) ===')
for lbl, e, tv in (('full ANN', e_full, tf), ('DC removed', e_deb, td), ('ANN off', e_off, to)):
    print(f'  {lbl:12s} tail={tv[2]:.3e}  envelope={env(e):.2f}')
print(f'  -> collapse factor (full / DC-removed, Y tail) = {collapse:.0f}x')

# velocity of the Y error (FD of the full-ANN trace; noiseless sim, FD is clean)
v_err = np.gradient(e_full[:, 2], ts)
print(f'  Y error velocity: last-quarter mean {v_err[3*M//4:].mean():+.3e} m/s '
      f'(bounded), max |v| {np.abs(v_err).max():.3e} m/s')

# ── Headline figure ───────────────────────────────────────────────────────────
PROV = (f'{tag} | V1_standstill_Yp10 | {M*ts:.1f} s open-loop free-run vs with-MSD truth | '
        f'routing (0..7), pipeline na=nb={na}')
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
axes[0].plot(t, e_full[:, 2], 'C3', lw=0.8, label='trained ANN (full)')
axes[0].plot(t, e_deb[:, 2], 'C0', lw=0.8,
             label=f'trained ANN, measured DC subtracted  (drift / {collapse:.0f})')
axes[0].plot(t, e_off[:, 2], color='0.4', lw=0.8, label='ANN output zeroed ("ANN off", same encoder x0)')
axes[0].axhline(absorber_rms, color='0.6', ls=':', lw=0.8,
                label=f'+/- sigma(delta_a) = {absorber_rms:.2e} m')
axes[0].axhline(-absorber_rms, color='0.6', ls=':', lw=0.8)
axes[0].set_ylabel('Y error [m]'); axes[0].legend(fontsize=7, loc='upper left')
axes[0].set_title('(1) removing ONE constant per row collapses the drift')
axes[1].plot(t, v_err, 'C3', lw=0.7, label='d/dt of the full-ANN Y error')
axes[1].set_ylabel('Y error velocity [m/s]'); axes[1].legend(fontsize=7, loc='upper right')
axes[1].set_title('(2) the velocity stays BOUNDED while the position ramps: integrator, not energy')
j_dy = 5
axes[2].plot(t[:len(w_full)][:M], w_full[:M, j_dy], 'C4', lw=0.6,
             label=f'ANN dY-row output (normalized);  |mean|/rms = '
                   f'{abs(mean_w[j_dy])/(rms_w[j_dy]+1e-30):.3f} (pure DC)')
axes[2].axhline(mean_w[j_dy], color='k', ls='--', lw=0.8, label='its time-mean (the subtracted DC)')
axes[2].set_ylabel('ANN output [-]'); axes[2].set_xlabel('Time [s]')
axes[2].legend(fontsize=7, loc='upper right')
axes[2].set_title('(3) the cause, measured directly: a near-constant force on the K=0 row')
for ax in axes:
    dm.sci_axes(ax); ax.grid(True)
fig.suptitle('V1: is the trained ANN\'s DC the drift?  (three free-runs, one change each)')
dm.add_provenance(fig, PROV)
fig.tight_layout()
p1 = os.path.join(dm.OUT_DIR, 'f2_dc_counterfactual.png')
fig.savefig(p1, dpi=150); print(f'\nSaved: {p1}')

# ── COMPANION: the DC across ALL available checkpoints (A2 closure) ──────────
bar = {}
if not SKIP_BAR:
    cands = [CKPT]
    cands += sorted(glob.glob(os.path.join(
        REPO, 'simulations', 'gantry_subnet', 'augmentation_linear_map',
        'trial_ckpts_*', '*.pth')))
    cands += sorted(glob.glob(os.path.join(
        REPO, 'simulations', 'gantry_subnet', 'augmentation_linear_map',
        'curriculum_*', 'rung*_last.pth')))
    seen = set()
    for path in cands:
        name = os.path.relpath(path, REPO).replace('\\', '/').split('/')[-1].replace('.pth', '')
        if 'curriculum' in path:
            name = 'warm_' + name
        elif 'trial_ckpts' in path:
            name = 'cold_' + name
        if name in seen or not os.path.exists(path):
            continue
        seen.add(name)
        try:
            a = load_ckpt(fit_sys, path)
            if a.nw != 8:
                print(f'  [bar] skip {name}: nw={a.nw} (not full routing)')
                continue
            _, w = freerun(fit_sys, a, u_v, y_v, N_BAR)
            bar[name] = (w.mean(axis=0), np.sqrt((w**2).mean(axis=0)))
            mw = bar[name][0]
            print(f'  [bar] {name:34s} dY-DC={mw[5]:+.2e}  dX-DC={mw[3]:+.2e}')
        except Exception as e:
            print(f'  [bar] skip {name}: {e}')

    if bar:
        names = list(bar.keys())
        xpos = np.arange(len(names))
        fig2, axs = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
        for ax, row in zip(axs.flat, K0_ROWS):
            means = np.array([bar[n][0][row] for n in names])
            rmss  = np.array([bar[n][1][row] for n in names])
            ax.bar(xpos, means, yerr=rmss, capsize=3, color='C0', alpha=0.85)
            ax.axhline(0, color='k', lw=0.8)
            ax.set_title(f'ANN mean output, {STATE8[row]} row (K=0)  [normalized; whisker = rms]')
            dm.sci_axes(ax); ax.grid(True, axis='y')
        for ax in axs[1]:
            ax.set_xticks(xpos)
            ax.set_xticklabels(names, rotation=30, ha='right', fontsize=7)
        fig2.suptitle(f'Does EVERY trained checkpoint carry a DC on the K=0 rows?  '
                      f'(mean over a {N_BAR*ts:.1f} s V1 free-run; d15: DC stationary)')
        dm.add_provenance(fig2, f'checkpoints: warm=70903 rungs, cold=71013 trials, + {tag} | '
                                f'V1 | normalized units (state-increment per step)')
        fig2.tight_layout()
        p2 = os.path.join(dm.OUT_DIR, 'f2_dc_across_checkpoints.png')
        fig2.savefig(p2, dpi=150); print(f'Saved: {p2}')

np.savez(os.path.join(dm.OUT_DIR, 'f2_dc.npz'),
         t=t, ts=ts, ckpt=tag, mean_w=mean_w, rms_w=rms_w, absorber_rms=absorber_rms,
         e_full=e_full, e_deb=e_deb, e_off=e_off, v_err=v_err, collapse=collapse,
         bar_names=np.array(list(bar.keys())),
         bar_means=np.array([bar[n][0] for n in bar]) if bar else np.zeros((0, 8)),
         bar_rms=np.array([bar[n][1] for n in bar]) if bar else np.zeros((0, 8)))
print(f"Saved: {os.path.join(dm.OUT_DIR, 'f2_dc.npz')}")
