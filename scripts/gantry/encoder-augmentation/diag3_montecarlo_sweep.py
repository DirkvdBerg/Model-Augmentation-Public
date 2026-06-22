"""
diag3_montecarlo_sweep.py
--------------------------
Light Monte Carlo sweep over N random seeds of linear_encoder_init_aug.

Motivated by Hoekstra 2026 Section 4.4 (Monte Carlo analysis with 10 models,
different random realisations for components left random).

Checks across N_SEEDS:
  1. x_b NRMS std < 1e-6 across all seeds
     W^b is deterministic from A/B/C/D -- seed must not affect it.
     (std < 1e-6 rather than 0 because float32 norm accumulation on
     near-zero ref denominators can give ~1e-7 NRMS noise; direct
     x_b diff between seeds is exactly 0.)
  2. x_a values differ across seeds (std > 0)
     W^a is kaiming random -- each seed gives a different x_a.
  3. x_a RMS spread is meaningful (different seeds explore different directions).

Also runs Jan's linear_encoder_init (no augmentation) as reference to confirm
x_b NRMS is the same whether or not nx_aug is present.

No training is performed -- this is a pure init quality sweep.
Training convergence across seeds is left to full model diagnostics.

Saves: simulations/gantry_subnet/encoder/diag3_montecarlo_sweep.npz
       simulations/gantry_subnet/encoder/diag3_montecarlo_sweep.json

Usage:
    conda run -n GraduationProject python \\
        scripts/gantry/encoder-augmentation/diag3_montecarlo_sweep.py
"""

import os
import sys
import json
import numpy as np
import torch
import deepSI
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model_augmentation.utils.utils import normalize_linear_ss_matrices
from model_augmentation.fit_systems.pre_encoder import (
    linear_encoder_init,
    linear_encoder_init_aug,
)
from model_augmentation.systems.gantry_ss import P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize

# =============================================================================
# Configuration
# =============================================================================

NX_PHYS  = 6
NX_ANN   = 2
nu, ny   = 3, 3

FS_ORIG  = 20000
FS_NEW   = 4000
D        = FS_ORIG // FS_NEW
TS_NEW   = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32

na = 4 * NX_PHYS + 1   # = 25  HEURISTIC: Jan's rule of thumb
nb = na

_NB_WIN   = nb + 1
_NA_WIN   = na + 1
_WIN_START = max(_NB_WIN, _NA_WIN) - 1   # = 25

N_NODES  = 16
N_HIDDEN = 2
N_SEEDS  = 10
SEEDS    = list(range(N_SEEDS))

MA_FRAC        = 0.50
MULTISINE_BAND = 'narrowband'

_msd_dir = os.path.join('multisine', f'm{round(MA_FRAC * 100)}')
if MULTISINE_BAND == 'narrowband':
    _msd_dir = os.path.join(_msd_dir, 'narrowband')
TRAJ_DIR = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab', _msd_dir)

TRAIN_FILES = [
    'T1_Y_osc.mat', 'T2_X_sym_Y_sweep.mat', 'T3_X_sym_Y000.mat',
    'T4_X_sym_Y030.mat', 'T5_theta_Y_coupling.mat', 'T6_lissajous_XY.mat',
    'T7_full_MIMO.mat', 'T8_multi_amp.mat', 'T9_Y_sweep_repeated.mat',
    'T10_multi_axis_repeated.mat',
]
VAL_FILE = 'V1_osc_Y025.mat'

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet', 'encoder')
os.makedirs(OUT_DIR, exist_ok=True)

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']

# =============================================================================
# Data loading and normalization (pure-scaled, same as diag2)
# =============================================================================

def load_mat(filename):
    d     = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u     = d['u_total'][::D].astype(DTYPE_NP)
    y     = d['y'][::D].astype(DTYPE_NP)
    x_log = d['x_logical'][::D].astype(DTYPE_NP)
    return u, y, x_log


def compute_norm_pure_scaled(train_data):
    u_all = np.concatenate([u for u, _, _ in train_data])
    y_all = np.concatenate([y for _, y, _ in train_data])
    x_all = np.concatenate([x for _, _, x in train_data])
    return dict(
        std_u = u_all.std(axis=0).astype(DTYPE_NP) + 1e-8,
        std_y = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8,
        std_x = x_all.std(axis=0).astype(DTYPE_NP) + 1e-8,
        u_all=u_all, y_all=y_all, x_all=x_all,
    )


def compute_analytical_baseline(y, x_logical):
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)
    pos     = (P_inv_T @ y.T).T                      # THEORY: q = inv(P^T) y
    vel     = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * FS_NEW          # HEURISTIC: backward FD
    vel[0]  = vel[1]
    x_ana   = np.hstack([pos, vel])
    rms_err = np.sqrt(np.mean((x_ana - x_logical)**2, axis=0))
    rms_gt  = np.sqrt(np.mean(x_logical**2, axis=0))
    return rms_err / (rms_gt + 1e-12)


def make_windows(val_u, val_y, norm):
    u_norm = val_u / norm['std_u']
    y_norm = val_y / norm['std_y']
    u_wins = np.lib.stride_tricks.sliding_window_view(
        u_norm, (_NB_WIN, nu)).reshape(-1, _NB_WIN * nu)
    y_wins = np.lib.stride_tricks.sliding_window_view(
        y_norm, (_NA_WIN, ny)).reshape(-1, _NA_WIN * ny)
    return (torch.tensor(u_wins.copy(), dtype=DTYPE_PT),
            torch.tensor(y_wins.copy(), dtype=DTYPE_PT))


def run_encoder(encoder, u_t, y_t):
    encoder.eval()
    with torch.no_grad():
        return encoder(u_t, y_t).numpy()


def nrms(x_enc_phys, x_gt, norm):
    x_phys = x_enc_phys * norm['std_x']
    x_gt_a = x_gt[_WIN_START:_WIN_START + len(x_phys)]
    T = min(len(x_phys), len(x_gt_a))
    err = np.sqrt(np.mean((x_phys[:T] - x_gt_a[:T])**2, axis=0))
    ref = np.sqrt(np.mean(x_gt_a[:T]**2, axis=0))
    return err / (ref + 1e-12)


# =============================================================================
# Plotting
# =============================================================================

def plot_xb_nrms_across_seeds(nrms_aug_all, nrms_jan_all, nrms_ana, out_path):
    """x_b NRMS per seed per channel -- should be flat (deterministic W^b)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, STATE_NAMES)):
        aug_vals = [nrms_aug_all[s][i] for s in range(N_SEEDS)]
        jan_vals = [nrms_jan_all[s][i] for s in range(N_SEEDS)]

        ax.plot(SEEDS, aug_vals, 'r-o', ms=5, lw=1, label='enc_aug x_b')
        ax.plot(SEEDS, jan_vals, 'b--s', ms=4, lw=1, label='enc_jan')
        ax.axhline(nrms_ana[i], color='k', lw=1, ls=':', label='analytical')
        ax.set_title(name)
        ax.set_xlabel('seed')
        ax.set_ylabel('NRMS')
        ax.set_yscale('log')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle('x_b NRMS across seeds -- flat = W^b deterministic')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_xa_rms_across_seeds(xa_rms_all, out_path):
    """x_a RMS per seed -- should vary (random W^a kaiming)."""
    fig, axes = plt.subplots(1, NX_ANN, figsize=(10, 4))
    if NX_ANN == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        rms_vals = [xa_rms_all[s][j] for s in range(N_SEEDS)]
        ax.bar(SEEDS, rms_vals, color='tab:green', alpha=0.8)
        ax.set_xlabel('seed')
        ax.set_ylabel('x_a RMS (normalized)')
        ax.set_title(f'x_ann[{j}] RMS across seeds -- variation = W^a random')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('x_a activation at init -- kaiming random, varies across seeds')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_xa_trajectories(xa_traj_all, out_path):
    """x_a time traces per seed -- visual diversity check."""
    T_plot = 500
    t = np.arange(T_plot) / FS_NEW

    fig, axes = plt.subplots(NX_ANN, 1, figsize=(12, 3 * NX_ANN), sharex=True)
    if NX_ANN == 1:
        axes = [axes]

    cmap = plt.cm.tab10
    for j, ax in enumerate(axes):
        for s in range(N_SEEDS):
            xa = xa_traj_all[s][:T_plot, j]
            ax.plot(t, xa, color=cmap(s / N_SEEDS), lw=0.8,
                    label=f'seed {s}', alpha=0.7)
        ax.set_ylabel(f'x_ann[{j}] (norm)')
        ax.set_title(f'x_ann[{j}] at init -- 10 seeds, diversity from W^a kaiming')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=5, ncol=5, loc='upper right')

    axes[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    print('=' * 70)
    print(f'Diagnostic 3: Monte Carlo sweep -- N_SEEDS={N_SEEDS}')
    print(f'  NX_PHYS={NX_PHYS}  NX_ANN={NX_ANN}  na=nb={na}')
    print(f'  FS_NEW={FS_NEW} Hz')
    print('=' * 70)

    # ── Data ─────────────────────────────────────────────────────────────────
    print('\nLoading data...')
    train_data = [load_mat(f) for f in TRAIN_FILES]
    val_u, val_y, val_x = load_mat(VAL_FILE)

    norm = compute_norm_pure_scaled(train_data)
    nrms_ana = compute_analytical_baseline(val_y, val_x)

    # ── System matrices (deterministic, seed-independent) ─────────────────────
    Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)
    sys_data_tmp = deepSI.System_data(u=norm['u_all'], y=norm['y_all'])
    sys_data_tmp.x = norm['x_all']
    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        Ad, Bd, Cd_dt, Dd_dt, sys_data_tmp)

    # ── I/O windows (same for all seeds) ─────────────────────────────────────
    u_t, y_t = make_windows(val_u, val_y, norm)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    print(f'\nRunning {N_SEEDS} seeds...')
    print(f'  {"seed":>4s} | {"x_b max_diff vs s0":>20s} | '
          f'{"xa[0] RMS":>10s} | {"xa[1] RMS":>10s}')
    print('  ' + '-' * 55)

    nrms_aug_all  = []   # (N_SEEDS, NX_PHYS)
    nrms_jan_all  = []   # (N_SEEDS, NX_PHYS)
    xa_rms_all    = []   # (N_SEEDS, NX_ANN)
    xa_traj_all   = []   # (N_SEEDS, T, NX_ANN)
    xb_seed0      = None

    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)

        enc_aug = linear_encoder_init_aug(
            A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
            nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
            nx_aug=NX_ANN,
            n_nodes_per_layer=N_NODES, n_hidden_layers=N_HIDDEN,
            flag_linear_only=True,   # linear-only: isolates W^b and W^a, no net noise
        )
        enc_jan = linear_encoder_init(
            A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
            nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
            n_nodes_per_layer=N_NODES, n_hidden_layers=N_HIDDEN,
            flag_linear_only=True,
        )

        out_aug = run_encoder(enc_aug, u_t, y_t)   # (T, NX_PHYS+NX_ANN)
        out_jan = run_encoder(enc_jan, u_t, y_t)   # (T, NX_PHYS)

        x_b_aug = out_aug[:, :NX_PHYS]
        x_a_aug = out_aug[:, NX_PHYS:]

        if s == 0:
            xb_seed0 = x_b_aug.copy()

        xb_diff = np.abs(x_b_aug - xb_seed0).max()

        nrms_aug_s = nrms(x_b_aug, val_x, norm)
        nrms_jan_s = nrms(out_jan,  val_x, norm)
        xa_rms_s   = [float(np.sqrt(np.mean(x_a_aug[:, j]**2))) for j in range(NX_ANN)]

        nrms_aug_all.append(nrms_aug_s)
        nrms_jan_all.append(nrms_jan_s)
        xa_rms_all.append(xa_rms_s)
        xa_traj_all.append(x_a_aug)

        print(f'  {s:4d} | {xb_diff:>20.3e} | '
              + ' | '.join(f'{xa_rms_s[j]:10.4e}' for j in range(NX_ANN)))

    nrms_aug_all = np.array(nrms_aug_all)   # (N_SEEDS, NX_PHYS)
    nrms_jan_all = np.array(nrms_jan_all)
    xa_rms_all   = np.array(xa_rms_all)     # (N_SEEDS, NX_ANN)

    # ── Checks ────────────────────────────────────────────────────────────────
    print('\n--- Summary ---')
    checks = {}

    # Check 1: x_b NRMS std across seeds ≈ 0
    # Threshold 1e-6: x_b tensors are byte-identical across seeds (verified by
    # max_diff=0.000e+00 column), but float32 norm accumulation on near-zero
    # ref denominators (unobservable axes like q2) can give ~1e-7 NRMS std.
    xb_nrms_std = nrms_aug_all.std(axis=0)
    ok_det = xb_nrms_std.max() < 1e-6
    checks['xb_nrms_deterministic'] = 'PASS' if ok_det else 'FAIL'
    print(f'  [{"PASS" if ok_det else "FAIL"}] x_b NRMS std across seeds '
          f'max={xb_nrms_std.max():.2e}  (expect < 1e-6, W^b is deterministic)')

    # Check 2: x_b matches Jan's encoder across all seeds
    jan_diff = np.abs(nrms_aug_all - nrms_jan_all).max()
    ok_jan = jan_diff < 1e-8
    checks['xb_matches_jan'] = 'PASS' if ok_jan else 'FAIL'
    print(f'  [{"PASS" if ok_jan else "FAIL"}] x_b NRMS matches Jan encoder '
          f'max_diff={jan_diff:.2e}')

    # Check 3: x_a varies across seeds
    xa_rms_std = xa_rms_all.std(axis=0)
    ok_var = xa_rms_std.min() > 1e-4
    checks['xa_varies_across_seeds'] = 'PASS' if ok_var else 'FAIL'
    print(f'  [{"PASS" if ok_var else "FAIL"}] x_a RMS varies across seeds '
          f'std={xa_rms_std}  (expect > 1e-4, W^a is random)')

    # Check 4: x_a mean RMS is nonzero
    xa_rms_mean = xa_rms_all.mean(axis=0)
    ok_active = xa_rms_mean.min() > 1e-3
    checks['xa_active_at_init'] = 'PASS' if ok_active else 'FAIL'
    print(f'  [{"PASS" if ok_active else "FAIL"}] x_a mean RMS across seeds '
          f'{xa_rms_mean}  (expect > 1e-3, kaiming init active)')

    print(f'\n  Analytical NRMS (reference):')
    for i, name in enumerate(STATE_NAMES):
        print(f'    {name}: {nrms_ana[i]:.4e}')

    print(f'\n  x_b NRMS mean across seeds:')
    for i, name in enumerate(STATE_NAMES):
        print(f'    {name}: mean={nrms_aug_all[:, i].mean():.4e}  '
              f'std={nrms_aug_all[:, i].std():.2e}')

    n_pass = sum(v == 'PASS' for v in checks.values())
    n_fail = sum(v == 'FAIL' for v in checks.values())
    print(f'\n{"="*70}')
    print(f'  {n_pass}/{len(checks)} checks passed')
    if n_fail > 0:
        print(f'  FAILED: {[k for k, v in checks.items() if v == "FAIL"]}')
    else:
        print('  Monte Carlo sweep confirmed:')
        print('    W^b deterministic: x_b NRMS identical across all seeds')
        print('    W^a random: x_a RMS varies and is active across seeds')

    # ── Save ──────────────────────────────────────────────────────────────────
    npz_path = os.path.join(OUT_DIR, 'diag3_montecarlo_sweep.npz')
    np.savez_compressed(npz_path,
        nrms_aug_all = nrms_aug_all,
        nrms_jan_all = nrms_jan_all,
        xa_rms_all   = xa_rms_all,
        nrms_ana     = nrms_ana,
        state_names  = np.array(STATE_NAMES),
        seeds        = np.array(SEEDS),
        fs           = np.float32(FS_NEW),
    )
    print(f'\n  Saved: {npz_path}')

    json_path = os.path.join(OUT_DIR, 'diag3_montecarlo_sweep.json')
    with open(json_path, 'w') as f:
        json.dump(dict(
            config=dict(NX_PHYS=NX_PHYS, NX_ANN=NX_ANN, N_SEEDS=N_SEEDS,
                        na=na, nb=nb, FS_NEW=FS_NEW),
            checks=checks,
            xb_nrms_mean={n: float(nrms_aug_all[:, i].mean())
                          for i, n in enumerate(STATE_NAMES)},
            xb_nrms_std ={n: float(nrms_aug_all[:, i].std())
                          for i, n in enumerate(STATE_NAMES)},
            xa_rms_mean ={j: float(xa_rms_mean[j]) for j in range(NX_ANN)},
            xa_rms_std  ={j: float(xa_rms_std[j])  for j in range(NX_ANN)},
            nrms_ana    ={n: float(nrms_ana[i]) for i, n in enumerate(STATE_NAMES)},
        ), f, indent=2)
    print(f'  Saved: {json_path}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_xb_nrms_across_seeds(
        nrms_aug_all, nrms_jan_all, nrms_ana,
        os.path.join(OUT_DIR, 'diag3_xb_nrms_seeds.png'))

    plot_xa_rms_across_seeds(
        xa_rms_all,
        os.path.join(OUT_DIR, 'diag3_xa_rms_seeds.png'))

    plot_xa_trajectories(
        xa_traj_all,
        os.path.join(OUT_DIR, 'diag3_xa_trajectories.png'))

    if n_fail > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
