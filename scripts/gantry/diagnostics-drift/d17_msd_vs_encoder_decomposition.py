"""
d17_msd_vs_encoder_decomposition.py -- what is the baseline free-run "drift" on the
K=0 axes: the absorber (MSD), the encoder initial condition, or discretization?
And is the excitation informative? Answered on T1, with-MSD and no-MSD datasets.

CLOSED-LOOP NOTE (measured): the injected 130-180 Hz multisine is a force in the
closed loop. For the NO-MSD (nominal) plant the servo rejects it, so u_total ~ 0.
The MSD is a tuned-mass-damper: it puts an ANTI-RESONANCE (notch) in the plant at
150 Hz, which collapses the loop gain there, so the servo can no longer reject the
multisine and it leaks into u_total (~45 N RMS / 170 N peak on T1). Hence
u(with MSD) != u(no MSD): the two datasets have DIFFERENT recorded inputs, so the
no-MSD OUTPUT cannot be used as a floor. The decomposition therefore lives entirely
in the with-MSD frame (driven by u_w, referenced to y_w); the no-MSD data is used
ONLY for the input/excitation figure.

Decomposition (exact additive identity, all driven by u_w):
    E  =  R  +  enc_IC
  E      = sim_baseline(x0_enc,   u_w) - y_w   actual free-run error (the ANN must fix)
  R      = sim_baseline(x0_true,  u_w) - y_w   residual at true IC = the absorber (ANN target)
  enc_IC = sim_baseline(x0_enc,   u_w) - sim_baseline(x0_true, u_w)   pure encoder-IC effect
Reading: a Y ramp in enc_IC but NOT in R => the drift precursor is the encoder IC
(supervisor's "no encoder"); R is a zero-mean 150 Hz absorber residual, no ramp.
(Discretization sits inside R and is small; it is NOT isolable here because u_n has
no 150 Hz content -- that needs a fine-vs-coarse up_sample run under u_w.)

Figures: (1) excitation (u_w vs u_n time + spectra, delta_a spectrum),
         (2) decomposition STAGE (X1,X2,Y), (3) decomposition LOGICAL (X,Theta,Y).

Data: T1_standstill_Ym30.mat, with-MSD in <traj>/, no-MSD in <traj>/baseline/.
Run (a few minutes; N_WIN caps the length for a smoke test):
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d17_msd_vs_encoder_decomposition.py
Env: N_WIN. Outputs -> simulations/gantry_subnet/diagnostics/
"""
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

REPO   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
sys.path.insert(0, GANTRY)                       # gantry_dynamic + entry file
sys.path.insert(0, os.path.dirname(__file__))    # drift_common

import deepSI
import drift_common as dc
from gantry_dynamic.data import compute_normalization, load_datasets, traj_dir, _load_u, _resample_u
from gantry_dynamic.model import build_model, get_encoder_dims
from gantry_dynamic.diagnostics import encoder_init_state
from gantry_interconnect_dynamic import CFG as cfg   # exact run config (no training on import)

N_WIN = os.environ.get('N_WIN', None)
FILE  = 'T1_standstill_Ym30.mat'                 # only file present in the no-MSD folder


def load_aug_from_dir(directory, filename, cfg, need_absorber=False):
    """Load u (block-mean stage force), y (stage), x_logical (N,6), delta_a from a mat
    file in an explicit directory (mirrors gantry_dynamic.data.load_mat_aug)."""
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f'missing {path}')
    d = loadmat(path, squeeze_me=True)
    u = _resample_u(_load_u(d), cfg).astype(cfg.dtype_np)
    N = len(u)
    D = cfg.d
    y = d['y'][::D][:N].astype(cfg.dtype_np)
    x_logical = d['x_logical'][::D][:N].astype(cfg.dtype_np)          # (N,6) [X,Th,Y,dX,dTh,dY]
    delta_a = None
    if need_absorber and 'delta_a' in d:
        delta_a = d['delta_a'][::D][:N].astype(cfg.dtype_np)
    return u, y, x_logical, delta_a


def amp_spectrum(x, fs):
    """Single-sided amplitude spectrum (Hann-windowed)."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    w = np.hanning(N)
    X = np.abs(np.fft.rfft(x * w)) * 2.0 / w.sum()
    f = np.fft.rfftfreq(N, 1.0 / fs)
    return f, X


# ── Build the pipeline exactly as the real run (seed, data, norm, model) ─────
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
fit_sys = build_model(cfg.hp, cfg, data, norm)

na, nb, na_right, nb_right = get_encoder_dims(cfg.hp, cfg)
K0 = max(na, nb)
ts = cfg.ts_new
fs = cfg.fs_new_hz

# ── Load the two T1 datasets (with-MSD frame; no-MSD used for the input figure) ─
dir_wMSD  = traj_dir(cfg)
dir_noMSD = os.path.join(traj_dir(cfg), 'baseline')
u_w, y_w, xl_w, da_w = load_aug_from_dir(dir_wMSD,  FILE, cfg, need_absorber=True)
u_n, y_n, xl_n, _    = load_aug_from_dir(dir_noMSD, FILE, cfg, need_absorber=False)
Nc = min(len(u_w), len(u_n))
u_w, y_w, xl_w = u_w[:Nc], y_w[:Nc], xl_w[:Nc]
u_n = u_n[:Nc]
da_w = da_w[:Nc] if da_w is not None else None
absorber_rms = float(da_w.std()) if da_w is not None else float('nan')

fnames = ['F_X1', 'F_X2', 'F_Y']
print(f'\nLoaded {FILE}: N={Nc}')
print('=== recorded total force u_total: RMS / peak per channel ===')
print(f"  {'chan':6s} {'with-MSD RMS':>13s} {'with-MSD peak':>14s} {'no-MSD RMS':>12s} {'no-MSD peak':>12s}")
for c, nm in enumerate(fnames):
    print(f'  {nm:6s} {np.sqrt((u_w[:,c]**2).mean()):>13.3f} {np.abs(u_w[:,c]).max():>14.3f} '
          f'{np.sqrt((u_n[:,c]**2).mean()):>12.3f} {np.abs(u_n[:,c]).max():>12.3f}')
print(f'  absorber displacement RMS sigma(delta_a) = {absorber_rms:.3e} m')

# ── Encoder x0 (from the REAL with-MSD measurements) vs true x0 (with-MSD state) ─
sd_wMSD = deepSI.System_data(u=u_w, y=y_w, dt=ts)
x0_enc_norm = encoder_init_state(fit_sys, sd_wMSD, K0, na, nb, na_right, nb_right, cfg)
x0_enc  = x0_enc_norm * norm.std_x.flatten() + norm.x_mean.flatten()   # (6,) physical logical
x0_true = xl_w[K0].astype(np.float64)                                  # with-MSD true state at K0

names = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY']
print(f'\n=== x0 at K0={K0} ({K0*ts*1e3:.1f} ms), with-MSD ===')
print(f"  {'state':8s} {'encoder':>13s} {'true':>13s} {'enc-true':>12s}")
for i, nm in enumerate(names):
    print(f'  {nm:8s} {x0_enc[i]:>13.5e} {x0_true[i]:>13.5e} {x0_enc[i]-x0_true[i]:>12.3e}')
dv_enc = x0_enc[3:] - x0_true[3:]
pred_X = dc.tau_X * dv_enc[0]
pred_Y = dc.tau_Y * dv_enc[2]
print(f'  encoder velocity error dv (X,Th,Y) = {dv_enc}')
print(f'  -> predicted settled K=0 offset  tau_X*dvX={pred_X:+.3e} m   tau_Y*dvY={pred_Y:+.3e} m')

# ── Two baseline free-runs under u_w (encoder x0 and true x0) ─────────────────
Ntot = Nc - K0 if N_WIN is None else min(int(N_WIN), Nc - K0)
sl   = slice(K0, K0 + Ntot)
u_seg = u_w[sl]
t = np.arange(Ntot) * ts
t0 = time.time()
y_enc,  st_enc  = dc.simulate_baseline(x0_enc,  u_seg, ts, return_state=True)
y_true, st_true = dc.simulate_baseline(x0_true, u_seg, ts, return_state=True)
print(f'\ntwo baseline free-runs ({time.time()-t0:.0f}s, {Ntot*ts:.1f} s each)')

# ── Curves: E = R + enc_IC, in STAGE and LOGICAL coords (with-MSD frame) ──────
y_w_seg = y_w[sl]
xl_w_seg = xl_w[sl][:, :3]
stage = {
    'E':      y_enc  - y_w_seg,       # actual free-run error (encoder x0)
    'R':      y_true - y_w_seg,       # residual at true x0 = the absorber (ANN target)
    'enc_IC': y_enc  - y_true,        # pure encoder-IC effect (= E - R)
}
logical = {
    'E':      st_enc[:, :3]  - xl_w_seg,
    'R':      st_true[:, :3] - xl_w_seg,
    'enc_IC': st_enc[:, :3]  - st_true[:, :3],
}

# ── Numeric tail summary (last 20 %) ──────────────────────────────────────────
tail = slice(int(0.8 * Ntot), Ntot)
print('\n=== STAGE tail-mean [m] (last 20 %):  E = R + enc_IC ===')
print(f"  {'chan':6s} {'E (total)':>12s} {'R (absorber)':>13s} {'enc_IC':>12s}")
for c, lbl in enumerate(['X1', 'X2', 'Y']):
    print(f'  {lbl:6s} {stage["E"][tail,c].mean():>12.3e} '
          f'{stage["R"][tail,c].mean():>13.3e} {stage["enc_IC"][tail,c].mean():>12.3e}')
settle = 5.0 * dc.tau_Y
if Ntot * ts < settle:
    print(f'  !! window {Ntot*ts:.1f}s < 5*tau_Y={settle:.1f}s: Y may not be settled (use full length).')

# ── Figure 1: excitation (input + absorber response) ─────────────────────────
t_full = np.arange(Nc) * ts
fig1, ax = plt.subplots(3, 1, figsize=(12, 9))
# (a) F_Y time series: with-MSD vs no-MSD
ax[0].plot(t_full, u_w[:, 2], 'C3', lw=0.6, label='with MSD  (u_total F_Y)')
ax[0].plot(t_full, u_n[:, 2], 'C0', lw=0.6, label='no MSD  (u_total F_Y ~ 0)')
ax[0].set_ylabel('F_Y [N]'); ax[0].set_xlabel('Time [s]')
ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax[0].grid(True); ax[0].legend(fontsize=8, loc='upper right')
ax[0].set_title('(a) recorded total force F_Y: absorber breaks the loop cancellation')
# (b) u_total amplitude spectra (with MSD, 3 channels) + no-MSD F_Y
for c, (nm, col) in enumerate(zip(fnames, ['C1', 'C2', 'C3'])):
    fq, U = amp_spectrum(u_w[:, c], fs)
    ax[1].semilogy(fq, U + 1e-12, col, lw=0.8, label=f'with MSD {nm}')
fq, Un = amp_spectrum(u_n[:, 2], fs)
ax[1].semilogy(fq, Un + 1e-12, 'C0', lw=0.8, ls='--', label='no MSD F_Y')
ax[1].axvspan(130, 180, color='0.8', alpha=0.5, label='130-180 Hz multisine')
ax[1].axvline(150, color='k', ls=':', lw=0.8)
ax[1].set_xlim(0, min(300, fs / 2)); ax[1].set_ylabel('|u_total| [N]'); ax[1].set_xlabel('Frequency [Hz]')
ax[1].grid(True, which='both'); ax[1].legend(fontsize=7, loc='upper right')
ax[1].set_title('(b) u_total spectrum: the leaked force sits in the 130-180 Hz band')
# (c) delta_a amplitude spectrum
if da_w is not None:
    fq, Da = amp_spectrum(da_w, fs)
    ax[2].semilogy(fq, Da + 1e-18, 'C4', lw=0.9, label='delta_a (absorber displ.)')
    ax[2].axvline(150, color='k', ls=':', lw=0.8, label='150 Hz (absorber f_a)')
    ax[2].set_xlim(0, min(300, fs / 2)); ax[2].set_ylabel('|delta_a| [m]'); ax[2].set_xlabel('Frequency [Hz]')
    ax[2].grid(True, which='both'); ax[2].legend(fontsize=7, loc='upper right')
    ax[2].set_title('(c) absorber response: 150 Hz peak confirms it is excited')
fig1.suptitle(f'{FILE}: excitation -- why u(with MSD) != u(no MSD)')
fig1.tight_layout()
stem_exc = os.path.join(dc.OUT_DIR, 'd17_excitation')
fig1.savefig(stem_exc + '.png', dpi=150)
print(f'Saved: {stem_exc}.png')

# ── Figures 2-3: decomposition (E = R + enc_IC) in stage and logical coords ───
COLORS = {'E': 'C3', 'R': 'C2', 'enc_IC': 'C0'}
LABELS = {
    'E':      'E = free-run error @ encoder x0   (sim - y, with MSD)',
    'R':      'R = residual @ true x0   (= absorber, the ANN target)',
    'enc_IC': 'enc_IC = encoder-IC effect   (= E - R)',
}


def make_decomp_fig(curves, panel_labels, units, guide_channels, title, fname):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ch, (axc, lab, unit) in enumerate(zip(axes, panel_labels, units)):
        for key in ('E', 'R', 'enc_IC'):
            axc.plot(t, curves[key][:, ch], COLORS[key], lw=0.7,
                     label=LABELS[key] if ch == 0 else None)
        axc.axhline(0, color='k', lw=0.5)
        if ch in guide_channels and np.isfinite(absorber_rms):
            axc.axhline( absorber_rms, color='0.4', ls=':', lw=0.9,
                         label=(f'+/- sigma(delta_a) = {absorber_rms:.2e} m '
                                f'(absorber displ. RMS = residual to learn)') if ch == 0 else None)
            axc.axhline(-absorber_rms, color='0.4', ls=':', lw=0.9)
        axc.set_ylabel(f'{lab} [{unit}]')
        axc.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axc.grid(True)
    axes[0].legend(fontsize=7, loc='upper right')
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f'Saved: {fname}')
    return fig


stem_stage = os.path.join(dc.OUT_DIR, 'd17_decomp_stage')
stem_log   = os.path.join(dc.OUT_DIR, 'd17_decomp_logical')
make_decomp_fig(stage,   ['X1', 'X2', 'Y'],    ['m', 'm', 'm'],   {0, 1, 2},
                f'{FILE}: free-run error decomposition E = R + enc_IC (STAGE coords)',   stem_stage + '.png')
make_decomp_fig(logical, ['X', 'Theta', 'Y'],  ['m', 'rad', 'm'], {0, 2},
                f'{FILE}: free-run error decomposition E = R + enc_IC (LOGICAL coords)', stem_log + '.png')

np.savez(stem_stage + '.npz', t=t, K0=K0, ts=ts, fs=fs, absorber_rms=absorber_rms,
         x0_enc=x0_enc, x0_true=x0_true, dv_enc=dv_enc, pred_X=pred_X, pred_Y=pred_Y,
         u_w=u_w, u_n=u_n, da_w=(da_w if da_w is not None else np.zeros(0)),
         stage_E=stage['E'], stage_R=stage['R'], stage_encIC=stage['enc_IC'],
         log_E=logical['E'], log_R=logical['R'], log_encIC=logical['enc_IC'])
print(f'Saved: {stem_stage}.npz')
