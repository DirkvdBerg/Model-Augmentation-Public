"""Three-curve comparison of the BASELINE model in closed loop.

Each adjacent pair isolates exactly one difference, which is the point: a single figure that
mixes implementation, rate and precision cannot attribute a discrepancy to any of them.

  curve 1  MATLAB, 20 kHz, double, MATLAB's own Cfb        export_baseline_closedloop.m
  curve 2  Python, 20 kHz, double, MATLAB's exported Cfb   pipeline block, use_f64=True
  curve 3  Python,  4 kHz, float32, re-discretised Cfb     pipeline block, as trained

  1 vs 2   INDEPENDENT IMPLEMENTATION. Same rate, same controller matrices, same precision.
           Only the code differs: Simulink-derived discrete state space against
           Gantry_State_Block's RK4 through the LFR rational structure. This is the loop
           analogue of the L4ss controller gate, and the gap this session found: the
           controller alone is gated to 1.9e-16 and the loop wiring to 1.1e-12 m
           (p1_equivalence.py), but that used the TRUTH model, never the baseline.

  2 vs 3   RATE AND PRECISION. Both Python, both the same block. 20 kHz double with the
           controller that MADE the record, against 4 kHz float32 with the re-discretised
           controller that TRAINING uses. p2_rate_compare.py measured sigma_max(So) at 150 Hz
           rising 15.3 % and the phase margin dropping 3.6 degrees between these rates, so
           this gap is expected to be the large one and to be dominated by the rate, not the
           precision (baseline_drift_replay.py measured float32 in the controller at
           5.2e-08 m, an order below).

The loop is the residual form of loss_variants.py:136-138 in all three curves,
`u = u_data + Cfb (y_data - y_model)`, so the comparison is of the object variant B trains.

Curve 1 needs `baseline_closedloop_<record>.mat`, written by export_baseline_closedloop.m in
MATLAB. If it is absent the script still runs and plots curves 2 and 3, saying so.

Outputs `figures/baseline_matlab_compare.png` and `.pdf`. Reads only; modifies nothing.
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry', 'msd-offset'))
sys.path.insert(0, HERE)

from gantry_dynamic.data import load_datasets, compute_normalization

import closed_loop as CL
from loss_variants import controller_ss
from baseline_drift_replay import (make_cfg, build_phy_block, pipeline_rollout,
                                   CH, Y_OP, FIGDIR, RECORD_LABEL)

RECORD = 'V1_standstill_Yp10'      # data.py:161 -> val_data = val_list[0]


def run_python(fs_new, use_f64, ctrl, ctrl_dtype, tag):
    """Baseline model, closed loop, residual form, at the requested rate and precision."""
    cfg = make_cfg(fs_new=fs_new, use_f64=use_f64)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    # Sample 0, analytic Simulink IC. See the long note in baseline_drift_replay.main: K0 is
    # an ENCODER constraint and does not belong in a replay, and because it is a sample count
    # the same K0 is a different instant at each rate. Starting both rates at 0 makes them
    # aligned by construction, so no offset correction is needed below.
    sd = data.val_data
    u_norm = ((sd.u - np.asarray(norm.u_mean).flatten())
              / np.asarray(norm.std_u).flatten()).astype(cfg.dtype_np)
    y_data = sd.y.astype(float)
    phy = build_phy_block(cfg.hp, cfg, norm)
    print('  [%s] %g kHz, %s, N = %d, from sample 0'
          % (tag, 1e-3 * cfg.fs_new_hz, cfg.dtype_pt, len(u_norm)), flush=True)
    y = pipeline_rollout(phy, cfg, norm, CL.x0_for('baseline', Y_OP), u_norm, y_data,
                         ctrl=ctrl, ctrl_dtype=ctrl_dtype)
    return y, y_data, cfg.ts_new


def rms(e):
    return np.sqrt(np.mean(e ** 2, axis=0))


def main():
    # ---- curve 2: 20 kHz, double, MATLAB's exported Cfb -------------------------------
    Ac, Bc, Cc, Dc, Y_op_rec = CL.load_controller(RECORD)
    assert abs(Y_op_rec - Y_OP) < 1e-12, 'record Y_op %.3f != %.3f' % (Y_op_rec, Y_OP)
    print('curve 2: Python 20 kHz double, MATLAB Cfb (%d states)' % Ac.shape[0], flush=True)
    y2, y_data20, ts20 = run_python(None, True, (Ac, Bc, Cc, Dc), np.float64, 'c2')

    # ---- curve 3: 4 kHz, float32, re-discretised Cfb ----------------------------------
    print('curve 3: Python 4 kHz float32, re-discretised Cfb', flush=True)
    ctrl4 = controller_ss(Y_OP, 1.0 / 4e3)
    y3, y_data4, ts4 = run_python(4000, False, ctrl4, np.float32, 'c3')

    e2, e3 = y2 - y_data20, y3 - y_data4

    # ---- curve 1: MATLAB ---------------------------------------------------------------
    mp = os.path.join(HERE, 'baseline_closedloop_%s.mat' % RECORD)
    have_matlab = os.path.exists(mp)
    if have_matlab:
        M = loadmat(mp, squeeze_me=True)
        y1 = np.asarray(M['y_cl'], float)[:len(y2)]
        e1 = y1 - y_data20[:len(y1)]
        print('curve 1: MATLAB 20 kHz double, N = %d (from sample 0)' % len(y1))
    else:
        y1 = e1 = None
        print('curve 1: MISSING %s\n'
              '         run export_baseline_closedloop.m in MATLAB, then rerun this script.'
              % os.path.basename(mp))

    # ---- numbers ------------------------------------------------------------------------
    print('\nclosed-loop error against the system, rms [m]   [%s]' % ', '.join(CH))
    if e1 is not None:
        print('  1  MATLAB  20 kHz f64   %s' % ' '.join('%10.4e' % v for v in rms(e1)))
    print('  2  Python  20 kHz f64   %s' % ' '.join('%10.4e' % v for v in rms(e2)))
    print('  3  Python   4 kHz f32   %s' % ' '.join('%10.4e' % v for v in rms(e3)))

    if e1 is not None:
        d12 = y1 - y2[:len(y1)]
        print('\n1 vs 2  IMPLEMENTATION  max |dy| = %s m'
              % ' '.join('%.3e' % v for v in np.max(np.abs(d12), axis=0)))
    # Both rates start at sample 0, so they are aligned by construction and no offset
    # correction is needed. 20 kHz -> 4 kHz by point sampling: y is band-limited far below
    # 2 kHz (D-087, D-099).
    n = min(len(y2) // 5, len(y3))
    d23 = y2[:5 * n:5] - y3[:n]
    print('2 vs 3  RATE+PRECISION  max |dy| = %s m'
          % ' '.join('%.3e' % v for v in np.max(np.abs(d23), axis=0)))

    # ---- figure ---------------------------------------------------------------------------
    t20 = np.arange(len(e2)) * ts20
    t4 = np.arange(len(e3)) * ts4
    t23 = t20[:5 * n:5]
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 8.4))
    for j in range(3):
        ax = axes[j, 0]
        if e1 is not None:
            ax.plot(t20[:len(e1)], e1[:, j], lw=0.8, color='k',
                    label='1  MATLAB, 20 kHz, f64' if j == 0 else None)
        ax.plot(t20, e2[:, j], lw=0.8, color='tab:blue',
                label='2  Python, 20 kHz, f64' if j == 0 else None)
        ax.plot(t4, e3[:, j], lw=0.8, color='tab:orange', ls='--',
                label='3  Python, 4 kHz, f32 (as trained)' if j == 0 else None)
        ax.axhline(0.0, color='k', lw=0.5, ls=':')
        ax.set_ylabel('%s err [m]' % CH[j])
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        ax.grid(alpha=0.3)

        ax = axes[j, 1]
        if e1 is not None:
            ax.plot(t20[:len(d12)], d12[:, j], lw=0.8, color='tab:green',
                    label='1 - 2  implementation' if j == 0 else None)
        ax.plot(t23, d23[:, j], lw=0.8, color='tab:red',
                label='2 - 3  rate + precision' if j == 0 else None)
        ax.axhline(0.0, color='k', lw=0.5, ls=':')
        ax.set_ylabel('%s difference [m]' % CH[j])
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        ax.grid(alpha=0.3)
        if j == 0:
            axes[j, 0].set_title('baseline model, closed loop: error vs the system')
            axes[j, 0].legend(fontsize=8, loc='best')
            ax.set_title('pairwise differences')
            ax.legend(fontsize=8, loc='best')
    axes[2, 0].set_xlabel('time [s]')
    axes[2, 1].set_xlabel('time [s]')
    fig.suptitle('Baseline model (no MSD) in closed loop, three implementations.\n'
                 '1 vs 2 isolates the implementation; 2 vs 3 isolates rate and precision. %s.'
                 % RECORD_LABEL, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = os.path.join(FIGDIR, 'baseline_matlab_compare.%s' % ext)
        fig.savefig(p, dpi=150)
        print('wrote %s' % p)
    if not have_matlab:
        print('\nNOTE: curve 1 absent. Figure shows curves 2 and 3 only.')


if __name__ == '__main__':
    main()
