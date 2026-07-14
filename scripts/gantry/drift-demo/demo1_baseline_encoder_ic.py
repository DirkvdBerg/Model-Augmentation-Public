"""
demo1_baseline_encoder_ic.py -- F1 (plan doc §12): the encoder-IC decomposition.

CLAIM (caption-claim): the untrained free-run error is the encoder initial condition,
bounded, and only the stiffness-free axes keep it.

Decomposition on T1 (with-MSD frame, exact additive identity, all driven by the recorded u):
    E      = sim_baseline(x0_enc,  u) - y     actual free-run error (encoder x0)
    R      = sim_baseline(x0_true, u) - y     residual at TRUE x0 (= absorber + replay, ANN target)
    enc_IC = E - R                            pure encoder-IC effect (two sims differing only in x0)
Certainty logic (C2 + C3): R is the one-variable counterfactual; enc_IC plateauing AT the
independently predicted tau_X*dvX line is the prediction match (measure + predict + remove).
Theta rings down (sprung) while X/Y park (K=0): the stiffness contrast (framing fact).

Figures -> simulations/gantry_subnet/diagnostics/drift-demo/:
  f1_encoder_ic_stage.png    X1/X2/Y   [m]      (what the sensors measure)
  f1_encoder_ic_logical.png  X/Theta/Y [m,rad,m] (yaw isolated; tau*dv line on the X panel ONLY)
  f1_encoder_ic.npz          all traces + x0s + predictions

Run (full 12 s, ~5 min of simulation; N_WIN caps for a smoke):
  conda run -n GraduationProject python scripts/gantry/drift-demo/demo1_baseline_encoder_ic.py
Env: N_WIN (cap sim length in samples).
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo_common as dm
from demo_common import CFG, STATE_NAMES
from gantry_dynamic import oracle

FILE  = 'T1_standstill_Ym30.mat'
N_WIN = os.environ.get('N_WIN', None)

# ── Pipeline (slim) + data ────────────────────────────────────────────────────
fit_sys, norm, K0, na, nb, na_right, nb_right = dm.build_pipeline()
u_w, y_w, xl_w, da_w = dm.load_T(FILE, need_absorber=True)
absorber_rms = float(da_w.std())
ts = CFG.ts_new

# ── x0: encoder (pipeline, untrained) vs true (stored state) ─────────────────
x0_enc  = dm.encoder_x0(fit_sys, norm, u_w, y_w, K0, na, nb, na_right, nb_right)
x0_true = xl_w[K0].astype(np.float64)
dv      = x0_enc[3:] - x0_true[3:]
pred_X  = dm.tau_X * dv[0]          # THEORY: settled K=0 offset of a velocity error = tau*dv
pred_Y  = dm.tau_Y * dv[2]          # (Y stated for completeness; Y is cross-coupling dominated, §3c)

print(f'\n=== x0 at K0={K0} ({K0*ts*1e3:.1f} ms), {FILE} ===')
print(f"  {'state':8s} {'encoder':>13s} {'true':>13s} {'enc-true':>12s}")
for i, nm in enumerate(STATE_NAMES):
    print(f'  {nm:8s} {x0_enc[i]:>13.5e} {x0_true[i]:>13.5e} {x0_enc[i]-x0_true[i]:>12.3e}')
print(f'  velocity error dv (X,Th,Y) = {dv}')
print(f'  predicted settled offset: tau_X*dvX = {pred_X:+.3e} m   tau_Y*dvY = {pred_Y:+.3e} m'
      f'   (tau_X={dm.tau_X:.2f}s, tau_Y={dm.tau_Y:.2f}s)')

# ── Two baseline free-runs (differ ONLY in x0) ───────────────────────────────
Ntot = (len(u_w) - K0) if N_WIN is None else min(int(N_WIN), len(u_w) - K0)
sl   = slice(K0, K0 + Ntot)
t    = np.arange(Ntot) * ts
print(f'\nfree-running the baseline twice ({Ntot*ts:.1f} s each) ...')
y_enc,  st_enc  = dm.simulate_baseline(x0_enc,  u_w[sl], ts, return_state=True)
y_true, st_true = dm.simulate_baseline(x0_true, u_w[sl], ts, return_state=True)

# Oracle (FP + true MSD, D-097): same u_w, true 8-state x0, pipeline ts/up_sample
# (fairness rule). Model class = truth, so its residual F is the DISCRETIZATION floor.
# Sim-only diagnostic reference, NOT an acceptance bar (lessons: thresholds are data-derived).
print('free-running the ORACLE (FP + true MSD) once ...')
# HEURISTIC: CENTRAL-difference vdelta_a (np.gradient), O(Ts^2). OE-1 (plan §13, CLOSED
# 2026-07-14): the backward-FD O(Ts) seed left a -1.1e-4 settled Y offset in the oracle
# (5x absorber RMS); the central-diff seed collapses it to +5.5e-6. Same mechanism class
# as the encoder prong: differentiated-position velocity bias -> K=0 settled offset.
vda = np.gradient(da_w.astype(np.float64), ts)
x_aug = np.stack([da_w, vda], axis=1)
Ncut = K0 + Ntot
y_orc, _da_hat = oracle.oracle_open_loop(
    u_w[:Ncut], xl_w[:Ncut], x_aug[:Ncut], CFG,
    up_sample=CFG.hp['up_sample'], start_ix=K0)
F_stage = y_orc[K0:] - y_w[sl]
F_log   = dm.stage_pos_to_logical(y_orc[K0:]) - xl_w[sl][:, :3]

stage = {                                   # vs the with-MSD truth outputs
    'E':      y_enc  - y_w[sl],
    'R':      y_true - y_w[sl],
    'enc_IC': y_enc  - y_true,
    'F':      F_stage,
}
logical = {                                 # vs the with-MSD true logical positions
    'E':      st_enc[:, :3]  - xl_w[sl][:, :3],
    'R':      st_true[:, :3] - xl_w[sl][:, :3],
    'enc_IC': st_enc[:, :3]  - st_true[:, :3],
    'F':      F_log,
}

tail = slice(int(0.8 * Ntot), Ntot)
print(f'\n=== tail-mean (last 20%) [m]:  E = R + enc_IC;  F = discretization floor ===')
print(f"  {'chan':6s} {'E (total)':>12s} {'R (target)':>12s} {'enc_IC':>12s} {'F (oracle)':>12s}")
for c, lbl in enumerate(['X1', 'X2', 'Y']):
    print(f'  {lbl:6s} {stage["E"][tail, c].mean():>12.3e} '
          f'{stage["R"][tail, c].mean():>12.3e} {stage["enc_IC"][tail, c].mean():>12.3e} '
          f'{stage["F"][tail, c].mean():>12.3e}')
if Ntot * ts < 5.0 * dm.tau_X:
    print(f'  !! window {Ntot*ts:.1f}s < 5*tau_X={5*dm.tau_X:.1f}s: X offset NOT settled (smoke only).')

# ── Figures (§12 spec: legends, sigma band, tau*dv line on LOGICAL X only) ───
COLORS = {'E': 'C3', 'R': 'C2', 'enc_IC': 'C0', 'F': '0.3'}
LABELS = {
    'E':      'E = free-run error @ encoder x0   (sim - y, with-MSD)',
    'R':      'R = residual @ true x0   (= absorber + replay, the ANN target)',
    'enc_IC': 'enc_IC = E - R   (pure encoder-IC effect)',
    'F':      'F = oracle (FP + true MSD) @ true x0 - y   (discretization floor; sim-only reference)',
}
PROV = (f'{FILE} | untrained baseline (ANN identically 0) | pipeline encoder na=nb={na} '
        f'(K0={K0}) | evaluation: {Ntot*ts:.1f} s open-loop free-run vs with-MSD truth')


def decomp_fig(curves, panels, units, pred_line_panel, fname, title):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ch, (ax, lab, unit) in enumerate(zip(axes, panels, units)):
        for key in ('E', 'R', 'enc_IC', 'F'):
            ax.plot(t, curves[key][:, ch], COLORS[key], lw=0.7,
                    label=LABELS[key] if ch == 0 else None)
        ax.axhline(0, color='k', lw=0.5)
        ax.axhline( absorber_rms, color='0.4', ls=':', lw=0.9,
                    label=(f'+/- sigma(delta_a) = {absorber_rms:.2e} m '
                           '(absorber displacement RMS = residual to learn)') if ch == 0 else None)
        ax.axhline(-absorber_rms, color='0.4', ls=':', lw=0.9)
        if ch == pred_line_panel:
            ax.axhline(pred_X, color='C1', ls='--', lw=1.2,
                       label=f'predicted settled offset tau_X*dvX = {pred_X:+.2e} m '
                             '(from measured encoder velocity error)')
        ax.set_ylabel(f'{lab} [{unit}]')
        dm.sci_axes(ax)
        ax.grid(True)
    axes[0].legend(fontsize=7, loc='upper right')
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(title)
    dm.add_provenance(fig, PROV)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f'Saved: {fname}')


decomp_fig(stage, ['X1', 'X2', 'Y'], ['m', 'm', 'm'], pred_line_panel=None,
           fname=os.path.join(dm.OUT_DIR, 'f1_encoder_ic_stage.png'),
           title=f'{FILE}: untrained baseline free-run -- what makes up the error? (STAGE coords)')
decomp_fig(logical, ['X', 'Theta', 'Y'], ['m', 'rad', 'm'], pred_line_panel=0,
           fname=os.path.join(dm.OUT_DIR, 'f1_encoder_ic_logical.png'),
           title=f'{FILE}: untrained baseline free-run -- what makes up the error? (LOGICAL coords)')

np.savez(os.path.join(dm.OUT_DIR, 'f1_encoder_ic.npz'),
         t=t, K0=K0, ts=ts, na=na, absorber_rms=absorber_rms,
         x0_enc=x0_enc, x0_true=x0_true, dv=dv, pred_X=pred_X, pred_Y=pred_Y,
         tau_X=dm.tau_X, tau_Y=dm.tau_Y,
         stage_E=stage['E'], stage_R=stage['R'], stage_encIC=stage['enc_IC'],
         stage_F=stage['F'], log_F=logical['F'],
         log_E=logical['E'], log_R=logical['R'], log_encIC=logical['enc_IC'])
print(f"Saved: {os.path.join(dm.OUT_DIR, 'f1_encoder_ic.npz')}")
