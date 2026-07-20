"""v1c: LOOK at the signals, with-MSD vs no-MSD (Jan's literal ask, Theme A).

Companion to v1b (which computed the mean statistics). Jan's ask is visual: run
the system before and after adding the extra mass and LOOK at the raw time
traces: does every signal oscillate symmetrically back and forth around its
operating point, or does it sit or settle to one side?

Design (per the figure rules in tasks/lessons.md):
  - one physical state per panel; the only overlay is the SAME quantity under
    the two conditions (no-MSD green, with-MSD red) plus dashed mean lines;
  - RAW signals, autoscaled: for standstill records the autoscale lands on the
    ripple band around the operating level, so symmetry is directly visible;
  - full show: forces (stage), positions (stage AND logical), velocities
    (logical), absorber delta_a (with-MSD only); one standstill and one motion
    record, 8 figures total;
  - means are computed on the FULL record; traces are decimated for plotting
    only (20 kHz -> 2 kHz, far above the 150/180 Hz content, no visual alias).
    # HEURISTIC: plot decimation 10x, keeps >= 11 samples per 180 Hz cycle
  - known first-thought answer, stated on the force figure: the force RIPPLE
    differs by design (multisine bands 130-180 Hz with-MSD vs 1-7 Hz no-MSD,
    gtd_config.m; the absorber notch makes the servo leak the in-band multisine
    into u_total, d17). The question here is the MEAN, not the ripple.

Outputs (folder convention, README header):
  figures -> scripts/gantry/gantry-zero-mean/figures/v1c_<record>_{forces,positions,velocities,absorber}.png
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DIR_W = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory', 'augmentation')
DIR_N = os.path.join(DIR_W, 'baseline')
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

FS = 20e3                 # native rate [Hz]
DEC = 10                  # HEURISTIC: plot decimation (means use full data)
RECORDS = ['T1_standstill_Ym30.mat', 'T8_ysweep_xmix.mat']

C_N, C_W = 'tab:green', 'tab:red'


def load_record(directory, filename):
    d = loadmat(os.path.join(directory, filename), squeeze_me=True)
    y = np.asarray(d['y'], dtype=np.float64)            # (N,3) stage positions
    xl = np.asarray(d['x_logical'], dtype=np.float64)   # (N,6) logical states
    u = np.asarray(d['u_total'], dtype=np.float64)      # (N,3) stage forces
    da = np.asarray(d['delta_a'], dtype=np.float64) if 'delta_a' in d else None
    return y, xl, u, da


def panel(ax, t, s_n, s_w, title):
    """One state, both conditions, dashed full-record means, sci axes."""
    m_n, m_w = s_n.mean(), s_w.mean()
    ax.plot(t, s_n[::DEC], color=C_N, lw=0.4, alpha=0.7, label='no-MSD')
    ax.plot(t, s_w[::DEC], color=C_W, lw=0.4, alpha=0.7, label='with-MSD')
    ax.axhline(m_n, color=C_N, ls='--', lw=1.2)
    ax.axhline(m_w, color=C_W, ls='--', lw=1.2)
    ax.set_title(title, fontsize=10)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=True)
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.02, f'mean no-MSD {m_n:+.2e} | with-MSD {m_w:+.2e}',
            transform=ax.transAxes, fontsize=7, va='bottom',
            bbox=dict(fc='white', alpha=0.75, ec='none'))


def finish(fig, axes_last_row, fname, suptitle):
    for ax in axes_last_row:
        ax.set_xlabel('time [s]')
    handles, labs = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc='upper right', fontsize=9)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'saved {out}')


for rec in RECORDS:
    tag = rec.replace('.mat', '')
    y_w, xl_w, u_w, da_w = load_record(DIR_W, rec)
    y_n, xl_n, u_n, _ = load_record(DIR_N, rec)
    n = min(len(y_w), len(y_n))
    y_w, xl_w, u_w = y_w[:n], xl_w[:n], u_w[:n]
    y_n, xl_n, u_n = y_n[:n], xl_n[:n], u_n[:n]
    da_w = da_w[:n] if da_w is not None else None
    t = np.arange(n)[::DEC] / FS
    test = f'Do the raw signals oscillate symmetrically about their operating point?  ({tag})'

    # forces (stage frame, as recorded)
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for c, nm in enumerate(['F_X1 [N]', 'F_X2 [N]', 'F_Y [N]']):
        panel(axes[c], t, u_n[:, c], u_w[:, c], nm)
    finish(fig, [axes[-1]], f'v1c_{tag}_forces.png',
           f'{test}\nrecorded total force u_total. NOTE: the ripple differs BY DESIGN'
           ' (multisine 130-180 Hz with-MSD vs 1-7 Hz no-MSD); the question is the MEAN'
           ' (dashed lines).')

    # positions: stage (top) + logical (bottom)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for c, nm in enumerate(['X1 [m]', 'X2 [m]', 'Y [m]']):
        panel(axes[0][c], t, y_n[:, c], y_w[:, c], f'stage {nm}')
    for c, nm in enumerate(['X [m]', 'Theta [rad]', 'Y [m]']):
        panel(axes[1][c], t, xl_n[:, c], xl_w[:, c], f'logical {nm}')
    finish(fig, axes[1], f'v1c_{tag}_positions.png',
           f'{test}\npositions, stage (top) and logical (bottom); axes autoscale to the'
           ' signal, so standstill panels show the ripple around the operating level')

    # velocities (logical)
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for c, nm in enumerate(['dX [m/s]', 'dTheta [rad/s]', 'dY [m/s]']):
        panel(axes[c], t, xl_n[:, 3 + c], xl_w[:, 3 + c], nm)
    finish(fig, [axes[-1]], f'v1c_{tag}_velocities.png',
           f'{test}\nlogical velocities')

    # absorber (with-MSD only; the offset coordinate itself)
    fig, ax = plt.subplots(figsize=(13, 3.6))
    m = da_w.mean()
    ax.plot(t, da_w[::DEC], color=C_W, lw=0.4, alpha=0.8, label='with-MSD')
    ax.axhline(0.0, color='k', lw=0.8)
    ax.axhline(m, color=C_W, ls='--', lw=1.2)
    ax.set_title('delta_a [m] (absorber displacement about its L0 rest position;'
                 ' exists only with-MSD)', fontsize=10)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.04, f'mean {m:+.2e} | std {da_w.std():.2e}'
            f' | |mean|/std {abs(m) / da_w.std():.1e}',
            transform=ax.transAxes, fontsize=8, va='bottom',
            bbox=dict(fc='white', alpha=0.75, ec='none'))
    ax.set_xlabel('time [s]')
    fig.suptitle(test, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = os.path.join(FIG_DIR, f'v1c_{tag}_absorber.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'saved {out}')
