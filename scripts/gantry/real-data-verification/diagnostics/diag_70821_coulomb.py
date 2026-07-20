"""
diag_70821_coulomb.py
---------------------
Diagnostic re-plot of job 70821 (telica_split, 22 traj, e4000) to inspect what
causes the residual open-loop mismatch (~40-68% NRMSE that will not train out).

Hypothesis (supervisor): the baseline has only VISCOUS friction (C matrix); the
real stage also has COULOMB (dry) friction F_c*sign(qdot). The optimizer, having
no Coulomb term, inflates the viscous coefficients far past their physical
datasheet maxima to approximate the friction force at the dominant speed -- but
cannot match its shape, so the fit plateaus.

Evidence produced here:
  1. loss curve (from history)                         -> plateau
  2. open-loop trajectory overlay (measured vs sim)    -> reproduces server plot
  3. residual time series with velocity-sign shading   -> error flips with motion
  4. residual-vs-velocity scatter (hysteresis)         -> Coulomb signature

Reads the trained params from the .pt; re-simulates locally. Reads the Telica
.log files through the normal loader (same path the training script uses).

Run:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/diag_70821_coulomb.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Importing run_telica applies all the patches (Telica datasheet init, .log
# loader) at module level; it does NOT train (guarded by __main__).
import run_telica_param_recovery as rt

PT = os.path.join(rt._ROOT, 'simulations', 'server-output',
                  'lfr_param_recovery_telica_split_22traj_e4000_70821.pt')
OUT = os.path.join(rt._ROOT, 'simulations', 'diag_70821_coulomb')
os.makedirs(OUT, exist_ok=True)

dtype = rt.tr.DTYPE
AXES = ['X1', 'X2', 'Y']

# Trajectories to inspect: iter0 (feedback-dominated -> many velocity reversals).
SPECS = [
    {'id': 'T1a', 'file': 'train/xpos_-60_ypos-40/iter0.log', 'label': 'xpos_-60_ypos-40 iter0 (train)'},
    {'id': 'E1a', 'file': 'test/xpos_-135_ypos-120/iter0.log', 'label': 'xpos_-135_ypos-120 iter0 (test)'},
]


def load_block():
    d = torch.load(PT, map_location='cpu', weights_only=False)
    block = rt._lfr_pb.ParameterizedLFRBlock(RMSE_baseline=1.0).to(dtype=dtype)
    lp = d['best_log_params'] if d.get('best_log_params') is not None else d['log_params']
    with torch.no_grad():
        block.log_params.copy_(torch.as_tensor(lp, dtype=dtype))
    # sanity: recovered physical params match saved params_learned
    rec = block._recover_params().detach().numpy()
    saved = np.asarray(d['params_learned'], dtype=float)
    err = np.abs(rec - saved).max()
    print(f'[load_block] max|recovered-saved params| = {err:.2e}  (best_epoch={d.get("best_epoch")})')
    return block, d


def plot_loss(d):
    h = d['history']
    ep = [x['epoch'] for x in h]
    loss = [x['mse_loss'] for x in h]
    rr = [(x['epoch'], x['full_traj_rmse_m']) for x in h
          if x.get('full_traj_rmse_m') is not None]
    ep_r = [e for e, _ in rr]; rmse = [r for _, r in rr]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].semilogy(ep, loss, lw=1.0, color='tab:blue')
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Normalised MSE (train windows)')
    ax[0].set_title('Training loss'); ax[0].grid(True, which='both', alpha=0.3)
    ax[1].plot(ep_r, rmse, lw=1.0, color='tab:red', marker='.', ms=3)
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Full-traj OL RMSE [m]')
    ax[1].set_title('Full-trajectory open-loop RMSE'); ax[1].grid(True, alpha=0.3)
    fig.suptitle('Job 70821 -- training converges then plateaus (structural floor)')
    fig.tight_layout()
    p = os.path.join(OUT, 'loss_curve_70821.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('  saved', p)


def sim_traj(block, spec):
    log_path = os.path.join(rt._DATASET_ROOT, spec['file'])
    u, q1, fs = rt.load_telica_log(log_path, dtype=dtype)     # u (1,T,3), q1 (T,3)
    x0 = rt._build_state_traj_logical(q1[:2], rt._P.to(dtype), float(rt._ts), dtype)[:1]
    ts_tensor = torch.tensor(float(rt._ts), dtype=dtype)
    y = rt._run_no_grad(block, x0, u, ts_tensor).Y[0].detach()
    T = min(y.shape[0], q1.shape[0])
    y, q1 = y[:T].numpy(), q1[:T].numpy()
    diff = y - q1                                            # (T,3) [m]
    vel = np.gradient(q1, axis=0) * float(fs)               # (T,3) [m/s] measured velocity
    t = np.arange(T) / float(fs)
    return t, q1, y, diff, vel, float(fs)


def plot_traj_and_residual(spec, t, q1, y, diff, vel):
    # trajectory overlay
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, lbl in enumerate(AXES):
        axs[i].plot(t, q1[:, i], color='tab:blue', lw=0.8, label='Measured')
        axs[i].plot(t, y[:, i], color='tab:orange', lw=0.8, ls='--', label='FP model (sim)')
        rms = np.sqrt(np.mean(diff[:, i] ** 2))
        nrm = rms / (q1[:, i].std() + 1e-12) * 100
        axs[i].set_ylabel(f'{lbl} [m]')
        axs[i].set_title(f'{lbl}: RMS={rms:.2e} m  NRMSE={nrm:.0f}%', fontsize=9)
        axs[i].grid(True, alpha=0.3); axs[i].legend(loc='upper right', fontsize=8)
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle(f'Open-loop overlay -- {spec["label"]} [{spec["id"]}] -- job 70821')
    fig.tight_layout()
    p = os.path.join(OUT, f'traj_overlay_{spec["id"]}.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig); print('  saved', p)

    # residual with velocity-sign shading
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, lbl in enumerate(AXES):
        axs[i].plot(t, diff[:, i], color='tab:red', lw=0.7, label='residual (sim-meas)')
        axs[i].axhline(0, color='k', lw=0.5, ls='--')
        # shade where velocity > 0
        axs[i].fill_between(t, diff[:, i].min(), diff[:, i].max(),
                            where=(vel[:, i] > 0), color='tab:green', alpha=0.08,
                            label='v>0')
        axs[i].set_ylabel(f'{lbl} [m]'); axs[i].grid(True, alpha=0.3)
        if i == 0:
            axs[i].legend(loc='upper right', fontsize=8)
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle(f'Residual vs motion direction -- {spec["label"]} [{spec["id"]}]\n'
                 f'green = measured velocity > 0 (Coulomb -> residual jumps sign with v)')
    fig.tight_layout()
    p = os.path.join(OUT, f'residual_shaded_{spec["id"]}.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig); print('  saved', p)


def plot_residual_vs_velocity(spec, diff, vel):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, lbl in enumerate(AXES):
        v = vel[:, i]; r = diff[:, i]
        m = np.abs(v) > (0.02 * np.abs(v).max())   # drop near-zero-velocity dwell
        sc = axs[i].scatter(v[m], r[m], s=2, alpha=0.25, c=np.sign(v[m]),
                            cmap='coolwarm')
        axs[i].axvline(0, color='k', lw=0.6); axs[i].axhline(0, color='k', lw=0.6)
        axs[i].set_xlabel(f'{lbl} velocity [m/s]'); axs[i].set_ylabel(f'{lbl} residual [m]')
        axs[i].set_title(lbl, fontsize=10); axs[i].grid(True, alpha=0.3)
    fig.suptitle(f'Residual vs measured velocity -- {spec["label"]} [{spec["id"]}]\n'
                 f'A step in residual across v=0 (colour flip) = unmodelled Coulomb friction')
    fig.tight_layout()
    p = os.path.join(OUT, f'residual_vs_velocity_{spec["id"]}.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig); print('  saved', p)


if __name__ == '__main__':
    block, d = load_block()
    print('\n[loss curve]')
    plot_loss(d)
    for spec in SPECS:
        print(f'\n[{spec["id"]}] {spec["label"]}')
        t, q1, y, diff, vel, fs = sim_traj(block, spec)
        for i, lbl in enumerate(AXES):
            rms = np.sqrt(np.mean(diff[:, i] ** 2))
            nrm = rms / (q1[:, i].std() + 1e-12) * 100
            print(f'    {lbl}: RMS={rms:.4e} m  NRMSE={nrm:5.1f}%   '
                  f'|v|max={np.abs(vel[:, i]).max():.3f} m/s')
        plot_traj_and_residual(spec, t, q1, y, diff, vel)
        plot_residual_vs_velocity(spec, diff, vel)
    print('\nDone. Plots in', OUT)
