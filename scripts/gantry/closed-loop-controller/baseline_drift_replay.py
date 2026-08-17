"""Does closing the loop remove the baseline's offset/drift, for the plant TRAINING uses?

`a_closed_loop_models.py` already answers this for `msd-offset/plant.py` (`deriv6`, numpy
float64, one RK4 step per sample at 20 kHz): open-loop replay of the trajectory record walks to
about -700 um, closed loop is flat, NRMS ratio 5.4e-04.

That is not the plant the pipeline trains. The pipeline uses `Gantry_State_Block`
(`model_augmentation/fit_systems/blocks.py`), torch float32 by default, `up_sample` RK4 substeps
per step, at `cfg.fs_new` = 4 kHz, reaching M^-1 through the LFR rational structure rather than
a hand-coded cofactor solve.

The PHYSICS is identical. `plant.py` imports its parameters from `gantry_ss.py`, and expanding
`M(Y) = M0 + M1 Y + M2 Y^2` reproduces `deriv6`'s `a11, a12, a22, a23, a33` term for term. So this
script is NOT a model check. It is a NUMERICS check, and the reason it is worth running is that
the drift is an integration phenomenon: the baseline has two poles at exactly z = 1, so a force
error accumulates with no decay. Whether the loop removes it is sensitive to precision, rate and
substep count in a way that reading the code cannot settle.

The baseline is `Gantry_State_Block` used standalone, which is what
`gantry_dynamic/baselines.py:compute_baseline_fp_nrms` documents as "zero ANN contribution".
No ANN is built and nothing is gated.

Closed loop uses the residual form of `loss_variants.py:136-138`,

    u_model = u_data + Cfb (y_data - y_model),   controller state zero at the start,

which is exact because Cfb is linear. Verified equivalent to driving from `r` with the
controller state reconstructed from the record (Kessels Remark 5.4) to 1e-7 m against a 0.2 m
model-data gap, at both 20 kHz and 4 kHz.

UNITS. The pipeline is normalised; Cfb is physical, m -> N. The residual is denormalised, filtered,
and the resulting force renormalised with the same `norm` object the pipeline built.

Outputs three figures, `.png` and `.pdf` each:
  `figures/baseline_drift_outputs.*`    the three output signals, system against both replays
  `figures/baseline_closedloop_only.*`  the same, open-loop replay dropped
  `figures/baseline_drift_replay.*`     the corresponding errors, with ramp fractions
  `figures/baseline_ctrl_precision.*`   float64 against float32 in the CONTROLLER path
Reads only; modifies nothing.
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry', 'msd-offset'))
sys.path.insert(0, HERE)

from gantry_dynamic.config import RunConfig
from gantry_dynamic.data import load_datasets, compute_normalization
from model_augmentation.fit_systems.blocks import Gantry_State_Block

import plant as PL
import closed_loop as CL
from loss_variants import controller_ss

CH = ['X1', 'X2', 'Y']
FIGDIR = os.path.join(HERE, 'figures')

# data.py:28-31 VAL_FILES, with the frozen operating point of each from
# gtd_build_records.m:58-61. Y_op is per record (D-039) and MUST be used to build that
# record's controller: kappa varies about 40 % across Y in [-0.30, +0.30], so the wrong
# operating point is a real error, not a rounding one.
VAL_RECORDS = [
    ('V1_standstill_Yp10',  0.10, 'standstill, narrowband 130-180 Hz'),
    ('V2_aprbs_Ylow',      -0.22, 'motion profile (aprbs)'),
    ('V3_ysweep_Yp10',      0.10, 'oscillatory y-sweep'),
    ('V4_lissajous_Ym10',  -0.10, 'oscillatory lissajous'),
]
RECORD_IX = int(os.environ.get('BDR_RECORD', 0))
RECORD_NAME, Y_OP, RECORD_KIND = VAL_RECORDS[RECORD_IX]
RECORD_LABEL = '%s, Y_op %+.2f (%s)' % (RECORD_NAME, Y_OP, RECORD_KIND)
SUFFIX = '_%s' % RECORD_NAME     # figures are per record; never overwrite another set


def make_cfg(fs_new=4000, use_f64=False):
    """Production settings, matching train_variants.make_cfg (the config B would train under).

    Defaults reproduce that config exactly. `fs_new=None` disables downsampling (20 kHz) and
    `use_f64=True` switches to double; both are used by baseline_matlab_compare.py to build
    the 20 kHz double reference curve.
    """
    return RunConfig(
        mode='augmentation', encoder_init='linear_map', ann_activation='tanh',
        joint_estimation=False, param_init_detune=None, snr=None, seed=42,
        fs_orig=20000, fs_new=fs_new, stride=10, use_f64=use_f64, save_flag=False,
        nf_probe_print=False,
        nx_ann=2, ann_route_ix=(3, 4, 5, 7),
        n_nodes_per_layer=16, n_hidden_layers=2, up_sample=1,
        batch_size=256, lr=1e-7, epochs=1, nf_seconds=0.100,
        orth_beta=0.0, orth_observe=False,
    )


# ----------------------------------------------------------------------------
# pipeline baseline: Gantry_State_Block standalone
# ----------------------------------------------------------------------------

def build_phy_block(hp, cfg, norm):
    """Exactly baselines.py:73-77. Gantry_State_Block alone IS the zero-ANN baseline."""
    return Gantry_State_Block(
        Y_op=None, std_x=norm.std_x, std_u=norm.std_u,
        x_mean=norm.x_mean, u_mean=norm.u_mean, Ts=cfg.ts_new,
        up_sample=hp['up_sample'],
    ).to(cfg.dtype_pt).eval()


def pipeline_rollout(phy_block, cfg, norm, x0_phys, u_norm, y_data_phys, ctrl=None,
                     ctrl_dtype=np.float64):
    """Open loop (ctrl=None) or closed loop (residual form) with the pipeline's own block.

    Mirrors baselines.py:_fp_step: output from the CURRENT state, then advance.

    `ctrl_dtype` sets the precision of the CONTROLLER path only; the plant is always the
    pipeline block at `cfg.dtype_pt`. float64 (default) isolates the plant's precision.
    Passing `cfg.dtype_np` reproduces what training actually does: `loss_variants.attach`
    casts Cfb to `torch.float32` whenever `cfg.use_f64` is False, so both the controller
    matrices AND its state are single precision, with a pole at exactly z = 1 accumulating
    in single precision.
    """
    NX, nu = cfg.nx_phys, cfg.nu
    Cd_norm, Dd_np = norm.Cd_norm, norm.Dd_np
    ystd, y0 = norm.ystd, norm.y0
    std_u = np.asarray(norm.std_u).flatten()

    x = ((np.asarray(x0_phys, dtype=cfg.dtype_np) - np.asarray(norm.x_mean).flatten())
         / np.asarray(norm.std_x).flatten())

    if ctrl is not None:
        Ac, Bc, Cc, Dc = (np.asarray(M, dtype=ctrl_dtype) for M in ctrl)
        xc = np.zeros(Ac.shape[0], dtype=ctrl_dtype)

    N = len(u_norm)
    y_out = np.empty((N, 3))
    with torch.no_grad():
        for t in range(N):
            y_phys = (Cd_norm @ x + Dd_np @ u_norm[t]) * ystd + y0
            y_out[t] = y_phys

            u_t_norm = u_norm[t]
            if ctrl is not None:
                dy = np.asarray(y_data_phys[t] - y_phys, dtype=ctrl_dtype)   # [m]
                u_fb = Cc @ xc + Dc @ dy                        # [N]
                u_t_norm = u_t_norm + u_fb / std_u
                xc = Ac @ xc + Bc @ dy

            xt = torch.tensor(x, dtype=cfg.dtype_pt).view(1, NX, 1)
            ut = torch.tensor(np.asarray(u_t_norm, dtype=cfg.dtype_np),
                              dtype=cfg.dtype_pt).view(1, nu, 1)
            x = phy_block(torch.cat([xt, ut], dim=1)).view(NX).cpu().numpy()
    return y_out


# ----------------------------------------------------------------------------
# float64 reference: deriv6 from msd-offset/plant.py, same rate, same inputs
# ----------------------------------------------------------------------------

def deriv6_rollout(cfg, x0_phys, u_phys, y_data_phys, ctrl=None):
    ts = cfg.ts_new
    x = np.asarray(x0_phys, float).copy()
    Pt = PL.P_np.T
    if ctrl is not None:
        Ac, Bc, Cc, Dc = ctrl
        xc = np.zeros(Ac.shape[0])

    N = len(u_phys)
    y_out = np.empty((N, 3))
    for t in range(N):
        y = Pt @ x[:3]
        y_out[t] = y
        u = u_phys[t]
        if ctrl is not None:
            dy = y_data_phys[t] - y
            u = u + (Cc @ xc + Dc @ dy)
            xc = Ac @ xc + Bc @ dy
        ul = PL.P_np @ u
        k1 = PL.deriv6(x, ul)
        k2 = PL.deriv6(x + .5 * ts * k1, ul)
        k3 = PL.deriv6(x + .5 * ts * k2, ul)
        k4 = PL.deriv6(x + ts * k3, ul)
        x = x + (ts / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_out


# ----------------------------------------------------------------------------

def summarise(tag, err, ts):
    """rms, MEAN, slope and ramp fraction, per channel.

    The mean is reported explicitly because `ramp_fraction` cannot see it: it returns
    variance explained, and np.var subtracts the mean first, so a pure DC offset scores
    0.00 %. Reporting only the ramp fraction and the final sample, as an earlier version
    did, says nothing about whether the error is zero-mean.
    """
    ramp = CL.ramp_fraction(err, ts=ts)
    rms = np.sqrt(np.mean(err ** 2, axis=0))
    mean = np.mean(err, axis=0)
    t = np.arange(len(err)) * ts
    A = np.vstack([t, np.ones_like(t)]).T
    slope = np.array([np.linalg.lstsq(A, err[:, j], rcond=None)[0][0] for j in range(3)])
    print('  %-28s rms %s | mean %s | slope %s | ramp %s'
          % (tag,
             ' '.join('%9.3e' % v for v in rms),
             ' '.join('%+9.3e' % v for v in mean),
             ' '.join('%+9.2e' % v for v in slope),
             ' '.join('%5.1f%%' % (100 * v) for v in ramp)))
    return rms, ramp


def main():
    cfg = make_cfg()
    hp = cfg.hp
    print('loading datasets at %d Hz ...' % cfg.fs_new_hz, flush=True)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)

    # Start at sample 0, from the generator's own initial condition.
    #
    # An earlier version started at K0 = max(na, nb), copying compute_baseline_fp_nrms. That
    # was wrong for a REPLAY. K0 exists because the ENCODER needs na/nb samples of history
    # before it can emit a state estimate, and the pipeline applies the same K0 to its
    # baseline runs so baseline and encoder-initialised model are scored over one interval.
    # Nothing here uses the encoder: the state is handed in directly.
    #
    # Two things went wrong because of it. K0 is a SAMPLE COUNT, so the same K0 is a different
    # instant at each rate (0.85 ms at 20 kHz, 4.25 ms at 4 kHz), which misaligned the rate
    # comparison. And starting mid-record inherits an absorber already in motion, so the
    # open-loop offset picked up a contribution the baseline cannot represent:
    # plant.py:44 gives dY(inf) = (MA/cy) * vda(t0), and vda(K0) alone accounted for most of
    # the measured offset. The record begins at rest with a 0.5 s hold, so starting at 0
    # makes the offset develop from the model mismatch instead of from the start index.
    #
    # x0 is the analytic Simulink IC (closed_loop.x0_for), not val_x_logical[0], which
    # avoids D-087's one-sided finite-difference velocity at sample 0 entirely.
    sd = data.val_list[RECORD_IX]
    x0_phys = CL.x0_for('baseline', Y_OP)
    ts = cfg.ts_new
    std_u = np.asarray(norm.std_u).flatten()
    u_mean = np.asarray(norm.u_mean).flatten()

    u_norm = ((sd.u - u_mean) / std_u).astype(cfg.dtype_np)
    u_phys = sd.u.astype(float)
    y_data = sd.y.astype(float)
    N = len(u_norm)
    print('  record: %s' % RECORD_LABEL)
    print('  N = %d samples (%.2f s) from sample 0, ts = %.3e s' % (N, N * ts, ts))
    print('  x0 = %s (analytic Simulink IC)' % np.array2string(x0_phys, precision=3))

    ctrl = controller_ss(Y_OP, ts)
    print('  Cfb at %g Hz, %d states, Y_op = %.2f' % (1 / ts, ctrl[0].shape[0], Y_OP))

    phy = build_phy_block(hp, cfg, norm)
    print('  baseline block: %s, dtype %s, up_sample %d  (zero ANN contribution)'
          % (type(phy).__name__, cfg.dtype_pt, hp['up_sample']), flush=True)

    print('\nrollouts ...', flush=True)
    y_p_ol = pipeline_rollout(phy, cfg, norm, x0_phys, u_norm, y_data, ctrl=None)
    y_p_cl = pipeline_rollout(phy, cfg, norm, x0_phys, u_norm, y_data, ctrl=ctrl)
    y_d_ol = deriv6_rollout(cfg, x0_phys, u_phys, y_data, ctrl=None)
    y_d_cl = deriv6_rollout(cfg, x0_phys, u_phys, y_data, ctrl=ctrl)
    # Same pipeline plant, but the CONTROLLER at cfg precision: what training actually runs.
    y_p_cl32 = pipeline_rollout(phy, cfg, norm, x0_phys, u_norm, y_data, ctrl=ctrl,
                                ctrl_dtype=cfg.dtype_np)

    e_p_ol, e_p_cl = y_p_ol - y_data, y_p_cl - y_data
    e_d_ol, e_d_cl = y_d_ol - y_data, y_d_cl - y_data

    print('\nbaseline error against the record   [%s]' % ', '.join(CH))
    r_p_ol, k_p_ol = summarise('pipeline  open loop', e_p_ol, ts)
    r_p_cl, k_p_cl = summarise('pipeline  CLOSED loop', e_p_cl, ts)
    r_d_ol, _ = summarise('analysis f64 open loop', e_d_ol, ts)
    r_d_cl, _ = summarise('analysis f64 CLOSED loop', e_d_cl, ts)
    e_p_cl32 = y_p_cl32 - y_data
    summarise('pipeline CLOSED, f32 ctrl', e_p_cl32, ts)
    print('  controller precision effect: max |y(f32 ctrl) - y(f64 ctrl)| = %s m'
          % ' '.join('%.3e' % v for v in np.max(np.abs(y_p_cl32 - y_p_cl), axis=0)))

    print('\nclosed/open rms ratio   pipeline %s   deriv6 %s'
          % (' '.join('%8.2e' % v for v in r_p_cl / r_p_ol),
             ' '.join('%8.2e' % v for v in r_d_cl / r_d_ol)))
    print('pipeline vs deriv6, max abs difference [m]   open %s   closed %s'
          % (' '.join('%8.2e' % v for v in np.max(np.abs(y_p_ol - y_d_ol), axis=0)),
             ' '.join('%8.2e' % v for v in np.max(np.abs(y_p_cl - y_d_cl), axis=0))))

    # ---------------- figure 1: outputs ----------------
    # The record IS the system WITH the hidden MSD, run closed loop by the generator.
    # Both baseline curves have NO MSD; they differ only in whether the loop is closed.
    t = np.arange(N) * ts
    izoom = t >= (t[-1] - 2.0)
    fig1, ax1 = plt.subplots(3, 2, figsize=(13.0, 8.4))
    for j in range(3):
        for col, m in enumerate([np.ones_like(t, dtype=bool), izoom]):
            ax = ax1[j, col]
            ax.plot(t[m], y_data[m, j], lw=1.0, color='k',
                    label='system: WITH MSD, closed loop' if (j == 0 and col == 0) else None)
            ax.plot(t[m], y_p_ol[m, j], lw=0.9, color='tab:red',
                    label='baseline model (no MSD), open loop' if (j == 0 and col == 0) else None)
            ax.plot(t[m], y_p_cl[m, j], lw=0.9, color='tab:blue', ls='--',
                    label='baseline model (no MSD), closed loop' if (j == 0 and col == 0) else None)
            ax.set_ylabel('%s [m]' % CH[j])
            ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_title('full record' if col == 0 else 'last 2 s')
    ax1[0, 0].legend(fontsize=8, loc='best')
    ax1[2, 0].set_xlabel('time [s]')
    ax1[2, 1].set_xlabel('time [s]')
    fig1.suptitle('Outputs: the system (WITH the hidden MSD, generated closed loop) '
                  'against the\nbaseline model WITHOUT the MSD, replayed open loop and closed '
                  'loop. %s, %g kHz, %s.' % (RECORD_LABEL, 1e-3 / ts, cfg.dtype_pt), fontsize=10)
    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = os.path.join(FIGDIR, 'baseline_drift_outputs%s.%s' % (SUFFIX, ext))
        fig1.savefig(p, dpi=150)
        print('wrote %s' % p)

    # ---------------- figure 2: closed loop only ----------------
    # Same two systems, open-loop replay dropped: how well does a baseline model WITHOUT the
    # MSD reproduce the system WITH it, once the loop is closed?
    fig2, ax2 = plt.subplots(3, 2, figsize=(13.0, 8.4))
    for j in range(3):
        for col, m in enumerate([np.ones_like(t, dtype=bool), izoom]):
            ax = ax2[j, col]
            ax.plot(t[m], y_data[m, j], lw=1.0, color='k',
                    label='system: WITH MSD, closed loop' if (j == 0 and col == 0) else None)
            ax.plot(t[m], y_p_cl[m, j], lw=0.9, color='tab:blue', ls='--',
                    label='baseline model (no MSD), closed loop' if (j == 0 and col == 0) else None)
            ax.set_ylabel('%s [m]' % CH[j])
            ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_title('full record' if col == 0 else 'last 2 s')
    ax2[0, 0].legend(fontsize=8, loc='best')
    ax2[2, 0].set_xlabel('time [s]')
    ax2[2, 1].set_xlabel('time [s]')
    fig2.suptitle('Closed loop only: the system (WITH the hidden MSD) against the\n'
                  'baseline model (WITHOUT the MSD). %s, %g kHz, %s.'
                  % (RECORD_LABEL, 1e-3 / ts, cfg.dtype_pt), fontsize=10)
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ('png', 'pdf'):
        p = os.path.join(FIGDIR, 'baseline_closedloop_only%s.%s' % (SUFFIX, ext))
        fig2.savefig(p, dpi=150)
        print('wrote %s' % p)

    # ---------------- figure 3: errors ----------------
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 8.4), sharex=True)
    for j in range(3):
        for col, (ep, kk, title) in enumerate([
                (e_p_ol, k_p_ol, 'open loop'),
                (e_p_cl, k_p_cl, 'CLOSED loop')]):
            ax = axes[j, col]
            ax.plot(t, ep[:, j], lw=0.8, color='tab:orange',
                    label='baseline model' if j == 0 else None)
            ax.axhline(0.0, color='k', lw=0.5, ls=':')
            ax.set_ylabel('%s err [m]' % CH[j])
            ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
            ax.grid(alpha=0.3)
            ax.text(0.99, 0.04, 'ramp %.2f %%' % (100 * kk[j]), ha='right',
                    transform=ax.transAxes, fontsize=8)
            if j == 0:
                ax.set_title('baseline error, %s' % title)
                ax.legend(fontsize=8, loc='upper left')
    axes[2, 0].set_xlabel('time [s]')
    axes[2, 1].set_xlabel('time [s]')
    fig.suptitle('Baseline model (zero ANN) replayed against the system, %g kHz, %s.\n'
                 'Left: open loop on recorded u. Right: loop closed with the verified Cfb. '
                 'Ramp %% = share of the error explained by a straight line.'
                 % (1e-3 / ts, cfg.dtype_pt), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = os.path.join(FIGDIR, 'baseline_drift_replay%s.%s' % (SUFFIX, ext))
        fig.savefig(p, dpi=150)
        print('wrote %s' % p)

    # ---------------- figure 4: controller precision ----------------
    # Same baseline model, same pipeline plant block, closed loop both times. The ONLY
    # difference is the precision of the controller path, which is what training varies via
    # cfg.use_f64 (loss_variants.attach). The z = 1 pole accumulates with no leak, so single
    # precision in the controller state is the plausible failure mode.
    fig4, ax4 = plt.subplots(3, 2, figsize=(13.0, 8.4), sharex=True)
    dctrl = y_p_cl32 - y_p_cl
    for j in range(3):
        ax = ax4[j, 0]
        ax.plot(t, e_p_cl[:, j], lw=0.8, color='tab:blue',
                label='float64 controller' if j == 0 else None)
        ax.plot(t, e_p_cl32[:, j], lw=0.8, color='tab:orange', ls='--',
                label='float32 controller (as trained)' if j == 0 else None)
        ax.axhline(0.0, color='k', lw=0.5, ls=':')
        ax.set_ylabel('%s err [m]' % CH[j])
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        ax.grid(alpha=0.3)

        ax = ax4[j, 1]
        ax.plot(t, dctrl[:, j], lw=0.8, color='tab:red')
        ax.axhline(0.0, color='k', lw=0.5, ls=':')
        ax.set_ylabel('%s difference [m]' % CH[j])
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        ax.grid(alpha=0.3)
        ax.text(0.99, 0.05, 'max %.2e m' % np.max(np.abs(dctrl[:, j])), ha='right',
                transform=ax.transAxes, fontsize=8)
        if j == 0:
            ax4[j, 0].set_title('baseline model, closed loop: error vs the system')
            ax4[j, 0].legend(fontsize=8, loc='best')
            ax.set_title('float32 controller minus float64 controller')
    ax4[2, 0].set_xlabel('time [s]')
    ax4[2, 1].set_xlabel('time [s]')
    fig4.suptitle('Controller precision. Baseline model (no MSD), closed loop,\nwith the '
                  'CONTROLLER at float64 against float32. %s, %g kHz, plant %s.'
                  % (RECORD_LABEL, 1e-3 / ts, cfg.dtype_pt), fontsize=10)
    fig4.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ('png', 'pdf'):
        p = os.path.join(FIGDIR, 'baseline_ctrl_precision%s.%s' % (SUFFIX, ext))
        fig4.savefig(p, dpi=150)
        print('wrote %s' % p)


if __name__ == '__main__':
    main()
