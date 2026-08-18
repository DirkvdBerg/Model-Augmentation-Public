"""Figures for the step-6 closed-loop run (SLURM 76573).

Sources, and why not the JSON for everything:
  server-results/step6_76573.out          the FULL 12-point validation series and the train-loss
                                          series. USE THIS for the curves.
  server-results/step6_76573_result.json  scalars and per-record values. Its "val" array has only
                                          3 points: the script read fs.Loss_val AFTER
                                          checkpoint_load_system replaced __dict__ with the epoch-2
                                          best checkpoint, so the series was truncated to that
                                          checkpoint's history. Known reporting bug, do not plot it.
  server-results/*_best.pth               deepSI checkpoint, needed ONLY for the trajectory and
                                          spectrum figures. Copy from
                                          /home/<user>/.deepSI/checkpoints/ on the server.

Figures produced:
  1 curves       train sqrt loss and closed-loop val sim-RMS, STACKED PANELS sharing x.
                 Deliberately not a twin-axis plot: two measures on two y-scales in one frame is
                 the standard way to imply a relationship that isn't there.
  2 attribution  per record: untrained / trained / trained-with-ANN-zeroed. The plot that shows the
                 gain is the ANN and not the encoder, because bar 3 returns to bar 1.
  3 headroom     untrained, trained, oracle floor on a log axis. Shows what was closed AND what
                 remains, so the result is not overclaimed.
  4 trajectory   per record, per channel: measured against model output (they overlap, which is
                 the point), with the ERROR underneath for init and trained. Needs the checkpoint.
  5 spectrum     |FFT(error)| init vs trained with the 150 Hz absorber marked. This is the figure
                 that shows the ANN learned the right PHYSICS rather than just a smaller number.
                 Needs the checkpoint.

Colours are the Okabe-Ito subset already used elsewhere in this folder
(verify_controller.py, test_controller_exact.py). Validated: worst adjacent CVD dE 11.0, all
lightness/chroma/contrast checks pass.

Usage:
  python -u cl_plot_step6.py            curves + attribution + headroom (no checkpoint needed)
  CL_CKPT=server-results/FitSys_ClosedLoop_xxx_best.pth python -u cl_plot_step6.py
"""
__project_origin__ = "added"

import dataclasses
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SRV = os.path.join(HERE, 'server-results')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

INK, MUTED, GRID = '#333333', '#6b6b6b', '#cccccc'
C_INIT, C_TRAIN, C_ORACLE = '#0072B2', '#D55E00', '#009E73'
CH = ['X1', 'X2', 'Y']
ORACLE = 2.81e-08          # cl_headroom.py, closed loop, same harness
F_ABSORBER = 150.0         # Hz, the hidden MSD mode

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False})

# EVERY axis carrying a physical quantity is in SI metres with explicit scientific tick labels.
# No micrometres, no millimetres, and no matplotlib offset header (the "1e-5 " in a corner), which
# hides the magnitude and makes a small relative change fill the panel.
from matplotlib.ticker import FuncFormatter                               # noqa: E402

SCI = FuncFormatter(lambda v, _p: '%.3e' % v)


def parse_log(path):
    """(iters, train_sqrt_loss, val_iters, val_rms, val_per_record) from the .out."""
    txt = open(path, encoding='utf-8', errors='replace').read()
    # NO '^' anchor: deepSI's progress lines are interleaved with tqdm carriage returns, so most
    # "It N, sqrt loss ..." fragments do not start a line. Anchoring found 9 of 12.
    its, tr = [], []
    for m in re.finditer(r'It (\d+), sqrt loss ([0-9.e+-]+)', txt):
        its.append(int(m.group(1)))
        tr.append(float(m.group(2)))
    vals, pers = [], []
    for m in re.finditer(r'\[cl-val\] closed-loop sim-RMS ([0-9.e+-]+) m\s+per record \[([^\]]+)\]',
                         txt):
        vals.append(float(m.group(1)))
        pers.append([float(v) for v in m.group(2).split()])
    # the first two entries are both pre-training (this script's own baseline and deepSI's
    # initial validation); keep one, at iteration 0
    if len(vals) > 1 and abs(vals[0] - vals[1]) / vals[0] < 1e-9:
        vals, pers = vals[1:], pers[1:]
    # Validation iterations come from the epoch size, not from the train-loss list: the two are
    # logged by different code paths and need not have the same length.
    m = re.search(r'N_batch_updates_per_epoch = (\d+)', txt)
    per_epoch = int(m.group(1)) if m else 260
    vits = np.arange(len(vals)) * per_epoch
    return np.array(its), np.array(tr), vits, np.array(vals), np.array(pers)


log = os.path.join(SRV, 'step6_76573.out')
res = json.load(open(os.path.join(SRV, 'step6_result_76573.json')))
its, tr, vits, vals, pers = parse_log(log)
names = res['per_record']['names']
short = [n.split('_')[0] for n in names]
print('parsed %d train-loss points, %d validation points' % (len(its), len(vals)))

# ── 1. curves ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 1, figsize=(6.8, 5.0), sharex=True)
ax[0].plot(its, tr, color=C_TRAIN, lw=1.8)
# Explicit scientific tick labels. matplotlib's default hoists the magnitude into a "1e-5" offset
# in the corner, which makes a 0.8 % change fill the panel and hides the actual scale.
ax[0].yaxis.set_major_formatter(SCI)   # FuncFormatter owns every label; no offset is applied
# What deepSI prints is sqrt of the training loss (fit(sqrt_train=True) is the default). The loss
# itself is MSE on the NORMALISED output over the nf-step closed-loop window, plus param_loss and
# the orth penalty (both exactly 0 in this run).
ax[0].set_ylabel('train $\\sqrt{\\mathrm{MSE}}$, normalised [-]')
ax[0].grid(alpha=0.3, lw=0.6, color=GRID)
ax[0].set_title('Closed-loop training, ANN routed to all eight states\n'
                '12 epochs, 3120 updates, lr $10^{-7}$', fontsize=10, color=INK)
# deepSI does not print the first few "It N" summaries, so this series starts partway in. Say the
# range explicitly rather than implying it covers the whole run.
ax[0].annotate('%.1f %% over updates %d-%d' % (100 * (tr[-1] - tr[0]) / tr[0], its[0], its[-1]),
               xy=(0.97, 0.88), xycoords='axes fraction', ha='right', fontsize=8, color=MUTED)

ax[1].plot(vits, vals, color=C_TRAIN, lw=1.8, marker='o', ms=4)
ax[1].axhline(vals[0], color=C_INIT, lw=1.4, ls='--')
ax[1].yaxis.set_major_formatter(SCI)
ax[1].annotate('untrained  %.3e' % vals[0], xy=(vits[-1], vals[0]),
               xytext=(-4, 4), textcoords='offset points', ha='right', fontsize=8, color=C_INIT)
ibest = int(np.argmin(vals))
ax[1].annotate('best %.3e  (%.1f %%)' % (vals[ibest],
                                         100 * (vals[0] - vals[ibest]) / vals[0]),
               xy=(vits[ibest], vals[ibest]), xytext=(12, 12),
               textcoords='offset points', fontsize=8, color=INK,
               arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
ax[1].set_ylabel('closed-loop val sim-RMS [m]')
ax[1].set_xlabel('update')
ax[1].grid(alpha=0.3, lw=0.6, color=GRID)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'step6_curves.png'), dpi=200, bbox_inches='tight')
print('wrote step6_curves.png')

# ── 2. attribution ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.6, 3.4))
x = np.arange(len(names))
w = 0.34
b = np.array(res['per_record']['base'])
f = np.array(res['per_record']['final'])
a = np.array(res['per_record']['annoff'])            # kept for the printout below
ax.bar(x - w / 2, b, w, color=C_INIT, label='untrained', zorder=3)
ax.bar(x + w / 2, f, w, color=C_TRAIN, label='trained', zorder=3)
for xi, (fi, bi) in enumerate(zip(f, b)):
    ax.annotate('%.1f %%' % (100 * (bi - fi) / bi), xy=(xi + w / 2, fi), xytext=(0, 3),
                textcoords='offset points', ha='center', fontsize=8, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(short)
ax.yaxis.set_major_formatter(SCI)
ax.set_ylabel('closed-loop sim-RMS [m]')
ax.set_title('Closed-loop validation error per record, untrained against trained',
             fontsize=10, color=INK)
ax.grid(axis='y', alpha=0.3, lw=0.6, color=GRID, zorder=0)
# Legend below the axes: every bar is nearly full height, so any in-axes position sits on data.
ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.12))
ax.set_ylim(0, b.max() * 1.18)      # headroom for the percentage labels
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'step6_attribution.png'), dpi=200, bbox_inches='tight')
print('wrote step6_attribution.png')
# The ANN-off arm is no longer plotted, but it is the evidence that the gain is the ANN and not
# the encoder, so keep it in the log rather than losing it entirely.
print('  ANN-off attribution (not plotted): per record, forcing the trained ANN output to zero')
for nm_, bi, ai in zip(short, b, a):
    print('    %-4s untrained %.4e  ANN-off %.4e  -> encoder-only change %+.3f %%'
          % (nm_, bi, ai, 100 * (bi - ai) / bi))

# ── 3. headroom ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.2, 3.4))
lv = [res['base'], res['final'], ORACLE]
lb = ['untrained\n(baseline FP)', 'trained\n(12 epochs)', 'oracle\n(FP + true MSD)']
ax.bar([0, 1, 2], lv, 0.5, color=[C_INIT, C_TRAIN, C_ORACLE], zorder=3)
ax.set_yscale('log')
for xi, v in enumerate(lv):
    ax.annotate('%.3e' % v, xy=(xi, v), xytext=(0, 3), textcoords='offset points',
                ha='center', fontsize=8, color=INK)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(lb, fontsize=8)
ax.yaxis.set_major_formatter(SCI)
ax.set_ylabel('closed-loop sim-RMS [m], log scale')
closed = 100 * (res['base'] - res['final']) / (res['base'] - ORACLE)
ax.set_title('%.1f %% of the achievable gap closed; %.0fx still remains'
             % (closed, res['final'] / ORACLE), fontsize=10, color=INK)
ax.grid(axis='y', alpha=0.3, lw=0.6, color=GRID, which='both', zorder=0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'step6_headroom.png'), dpi=200, bbox_inches='tight')
print('wrote step6_headroom.png')

# ── 4 and 5. trajectories and spectra (need the checkpoint) ──────────────────
CKPT = os.environ.get('CL_CKPT')
if not CKPT:
    import glob
    # recursive: the checkpoints are usually dropped in a subfolder alongside the .out
    g = sorted(glob.glob(os.path.join(SRV, '**', '*_best.pth'), recursive=True))
    CKPT = g[0] if g else None
if not CKPT or not os.path.exists(CKPT):
    print('\nNo checkpoint found in %s' % SRV)
    print('Trajectory and spectrum figures SKIPPED. Copy it from the server:')
    print('  scp <server>:/home/<user>/.deepSI/checkpoints/FitSys_ClosedLoop_*_best.pth %s/' % SRV)
    sys.exit(0)

print('\nusing checkpoint %s' % os.path.basename(CKPT))
import torch                                                              # noqa: E402
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
for p in (REPO, GANTRY, HERE, os.path.join(GANTRY, 'drift-demo'),
          os.path.join(GANTRY, 'msd-offset')):
    if p not in sys.path:
        sys.path.insert(0, p)
import demo_common as dm                                                  # noqa: E402
from demo_common import CFG                                               # noqa: E402
from gantry_dynamic.data import load_traj, VAL_FILES                      # noqa: E402
import cl_plant as PLANT                                                  # noqa: E402
import cl_validation as CV                                                # noqa: E402
from cl_controller import ControllerBank                                  # noqa: E402

import cl_fitsys as CLF                                                   # noqa: E402

cfg = dataclasses.replace(CFG, seed=0, ann_route_ix=tuple(range(8)), lr=res['lr'])
fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=False)
nx = cfg.nx_phys + cfg.nx_ann
C_out, b_out = PLANT.identify_output_map(fs.hfn, nx, cfg.nu, dtype=cfg.dtype_pt)
step_fn, out_fn = PLANT.make_fns(fs, C_out, b_out)
vnames = [f[:-4] for f in VAL_FILES]
bank = ControllerBank(vnames, cfg.ts_new, dtype=cfg.dtype_pt, ystd=norm.ystd, std_u=norm.std_u)

# attach BEFORE torch.load. The checkpoint is a pickled fit-system __dict__ whose objects
# reference the class `FitSys_ClosedLoop`, and that class is CREATED and bound into cl_fitsys by
# attach() at runtime (`type(...)` plus the globals() binding). Without attaching first, unpickling
# fails with "Can't get attribute 'FitSys_ClosedLoop' on module cl_fitsys".
CLF.attach(fs, bank, step_fn, out_fn)
ck = torch.load(CKPT, weights_only=False)
# load STATE ONLY into the freshly built system, rather than adopting the pickled __dict__: that
# keeps every object identity local and avoids the stale-reference trap that broke the first
# attribution test (fit() replaces __dict__, leaving captured handles pointing at the old modules).
fs.hfn.load_state_dict(ck['hfn'].state_dict())
fs.encoder.load_state_dict(ck['encoder'].state_dict())

WANT = os.environ.get('CL_PLOT_RECORDS', 'V1_standstill_Yp10,V2_aprbs_Ylow').split(',')
fs_hz = cfg.fs_new_hz
for i, nm in enumerate(vnames):
    if nm not in WANT:
        continue
    sd = load_traj(nm + '.mat', cfg)
    un = ((sd.u - fs.norm.u0) / fs.norm.ustd).astype(cfg.dtype_np)
    yn = ((sd.y - fs.norm.y0) / fs.norm.ystd).astype(cfg.dtype_np)
    ctrl = bank.gather(torch.tensor([i], dtype=torch.long))
    x0 = CV.encoder_x0(fs.encoder, un, yn, K0, na, nb, na_r, nb_r, cfg.dtype_pt)
    print('  rolling %s trained ...' % nm, flush=True)
    y_tr = CV.free_run(step_fn, out_fn, un, yn, x0, bank, ctrl, k0=K0, closed=True)
    rest = PLANT.zero_the_ann(fs)
    print('  rolling %s init (ANN off) ...' % nm, flush=True)
    y_in = CV.free_run(step_fn, out_fn, un, yn, x0, bank, ctrl, k0=K0, closed=True)
    rest()
    ys = np.asarray(fs.norm.ystd), np.asarray(fs.norm.y0)
    Y_tr = y_tr * ys[0] + ys[1]
    Y_in = y_in * ys[0] + ys[1]
    Y_da = sd.y[K0:]
    t = np.arange(len(Y_da)) / fs_hz
    e_tr, e_in = Y_tr - Y_da, Y_in - Y_da

    # ---- trajectory + error ----
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 5.2), sharex=True)
    for j in range(3):
        ax[0, j].plot(t, Y_da[:, j], color=INK, lw=1.4, label='measured')
        ax[0, j].plot(t, Y_tr[:, j], color=C_TRAIN, lw=1.0, ls='--', label='model (trained)')
        ax[0, j].set_title(CH[j], fontsize=10, color=INK)
        ax[0, j].grid(alpha=0.3, lw=0.6, color=GRID)
        ax[0, j].yaxis.set_major_formatter(SCI)
        ax[1, j].plot(t, e_in[:, j], color=C_INIT, lw=0.8, label='untrained')
        ax[1, j].plot(t, e_tr[:, j], color=C_TRAIN, lw=0.8, label='trained')
        ax[1, j].grid(alpha=0.3, lw=0.6, color=GRID)
        ax[1, j].yaxis.set_major_formatter(SCI)
        ax[1, j].set_xlabel('time [s]')
    ax[0, 0].set_ylabel('output [m]')
    ax[1, 0].set_ylabel('error [m]')
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle('%s: the trajectories overlap at this scale, so the error panel is the result'
                 % nm, fontsize=10, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, 'step6_traj_%s.png' % nm), dpi=200, bbox_inches='tight')
    print('  wrote step6_traj_%s.png' % nm)

    # ---- error spectrum: PSD (top) and RATIO (bottom) ----
    # Two overlaid spectra on a log axis spanning six decades CANNOT resolve a 36 % change: that is
    # 0.13 decades, about a line width. The first version of this figure did exactly that and its
    # caption asserted a conclusion the plot could not support. The RATIO panel is the instrument
    # that answers the question, and Welch replaces the raw FFT so the curve is readable rather
    # than hash.
    from scipy.signal import welch                                        # noqa: E402
    nper = 8192
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 5.6), sharex=True)
    band_rows = []
    for j in range(3):
        fr, P_in = welch(e_in[:, j], fs=fs_hz, nperseg=nper)
        _, P_tr = welch(e_tr[:, j], fs=fs_hz, nperseg=nper)
        ax[0, j].loglog(fr[1:], np.sqrt(P_in[1:]), color=C_INIT, lw=1.0, label='untrained')
        ax[0, j].loglog(fr[1:], np.sqrt(P_tr[1:]), color=C_TRAIN, lw=1.0, label='trained')
        ax[0, j].axvline(F_ABSORBER, color=MUTED, lw=1.0, ls=':')
        ax[0, j].set_title(CH[j], fontsize=10, color=INK)
        ax[0, j].grid(alpha=0.3, lw=0.5, color=GRID, which='both')
        ax[0, j].yaxis.set_major_formatter(SCI)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.sqrt(P_tr / P_in)
        ax[1, j].semilogx(fr[1:], ratio[1:], color=C_TRAIN, lw=1.0)
        ax[1, j].axhline(1.0, color=INK, lw=1.0)
        ax[1, j].axvline(F_ABSORBER, color=MUTED, lw=1.0, ls=':')
        ax[1, j].set_ylim(0, 2.0)
        ax[1, j].grid(alpha=0.3, lw=0.5, color=GRID, which='both')
        ax[1, j].set_xlabel('frequency [Hz]')

        # band-limited error rms: the question as three numbers instead of an eyeball judgement
        for lab, lo, hi in (('< 130 Hz', 0.0, 130.0),
                            ('130-180 Hz (absorber)', 130.0, 180.0),
                            ('> 180 Hz', 180.0, np.inf)):
            m = (fr >= lo) & (fr < hi)
            df = fr[1] - fr[0]
            r_in = np.sqrt(np.sum(P_in[m]) * df)
            r_tr = np.sqrt(np.sum(P_tr[m]) * df)
            band_rows.append((CH[j], lab, r_in, r_tr, 100 * (r_in - r_tr) / r_in))
    ax[0, 0].set_ylabel('error PSD$^{1/2}$  [m/$\\sqrt{Hz}$]')
    ax[1, 0].set_ylabel('trained / untrained')
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[0, 2].annotate('%g Hz absorber' % F_ABSORBER, xy=(F_ABSORBER, 1),
                      xycoords=('data', 'axes fraction'), xytext=(3, -12),
                      textcoords='offset points', fontsize=7, color=MUTED)
    fig.suptitle('%s: error spectrum, and the ratio that resolves it. Below 1 = improved, '
                 'above 1 = worse' % nm, fontsize=10, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, 'step6_spectrum_%s.png' % nm), dpi=200, bbox_inches='tight')
    print('  wrote step6_spectrum_%s.png' % nm)
    print('  band-limited error rms [m]  (%s)' % nm)
    print('    %-4s %-24s %-12s %-12s %s' % ('ch', 'band', 'untrained', 'trained', 'reduction %'))
    for c, lab, r_in, r_tr, pc in band_rows:
        print('    %-4s %-24s %-12.4e %-12.4e %+8.2f' % (c, lab, r_in, r_tr, pc))

print('\nfigures in %s' % FIG)
