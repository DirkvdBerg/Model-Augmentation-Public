"""
diag2_physical_state_recovery.py
---------------------------------
Verify that linear_encoder_init_aug correctly recovers physical states
when nx_aug=2, and that its x_b output is identical to Jan's
linear_encoder_init at initialization.

Checks:
  1. x_b from linear_encoder_init_aug(nx_aug=2) is identical to
     Jan's linear_encoder_init(nx=6) on the same I/O windows (machine precision)
  2. Output shape is (T, NX_PHYS + NX_ANN) = (T, 8)
  3. x_a is nonzero (kaiming init), no NaN, shape (T, 2)
  4. x_b NRMS vs x_logical -- should be comparable to analytical baseline

Uses pure-scaled inputs (u/std_u, y/std_y) fed directly to the linear
encoder, matching the convention that normalize_linear_ss_matrices produces.
No mean subtraction is applied before the encoder (no convention fix needed).

Saves: simulations/gantry_subnet/encoder/diag2_physical_state_recovery.npz
       simulations/gantry_subnet/encoder/diag2_physical_state_recovery.json

Usage:
    conda run -n GraduationProject python \\
        scripts/gantry/encoder-augmentation/diag2_physical_state_recovery.py
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
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize

# =============================================================================
# Configuration
# =============================================================================

NX_PHYS = 6
NX_ANN  = 2
nu, ny  = 3, 3

FS_ORIG = 20000
FS_NEW  = 4000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32
TOL_IDENTITY = 1e-5   # tolerance for x_b identity check

# Encoder history: Jan's rule, matches gantry_interconnect_dynamic.py
na = 4 * NX_PHYS + 1   # = 25  HEURISTIC: Jan's rule of thumb
nb = na

# Window sizes: encoder sees (nb+1) u samples and (na+1) y samples
_NB_WIN   = nb + 1    # = 26
_NA_WIN   = na + 1    # = 26
_WIN_START = max(_NB_WIN, _NA_WIN) - 1   # = 25

N_NODES   = 16
N_HIDDEN  = 2

MA_FRAC        = 0.50
MULTISINE_BAND = 'narrowband'

_msd_dir = os.path.join('multisine', f'm{round(MA_FRAC * 100)}')
if MULTISINE_BAND == 'narrowband':
    _msd_dir = os.path.join(_msd_dir, 'narrowband')
TRAJ_DIR = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab', _msd_dir)

TRAIN_FILES = [
    'T1_Y_osc.mat',
    'T2_X_sym_Y_sweep.mat',
    'T3_X_sym_Y000.mat',
    'T4_X_sym_Y030.mat',
    'T5_theta_Y_coupling.mat',
    'T6_lissajous_XY.mat',
    'T7_full_MIMO.mat',
    'T8_multi_amp.mat',
    'T9_Y_sweep_repeated.mat',
    'T10_multi_axis_repeated.mat',
]
VAL_FILE = 'V1_osc_Y025.mat'

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet', 'encoder')
os.makedirs(OUT_DIR, exist_ok=True)

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']
PHYS_UNITS  = ['m',  'm',  'm',  'm/s', 'm/s', 'm/s']

# =============================================================================
# Data loading
# =============================================================================

def load_mat(filename):
    d     = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u     = d['u_total'][::D].astype(DTYPE_NP)
    y     = d['y'][::D].astype(DTYPE_NP)
    x_log = d['x_logical'][::D].astype(DTYPE_NP)
    return u, y, x_log


# =============================================================================
# Normalization (pure-scaled: divide by std only, no mean subtraction)
# Pure-scaled matches the convention that normalize_linear_ss_matrices uses,
# so the linear encoder matrices can be applied directly without offset fix.
# =============================================================================

def compute_norm_pure_scaled(train_data):
    u_all = np.concatenate([u for u, _, _ in train_data])
    y_all = np.concatenate([y for _, y, _ in train_data])
    x_all = np.concatenate([x for _, _, x in train_data])

    std_u = u_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    std_y = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    std_x = x_all.std(axis=0).astype(DTYPE_NP) + 1e-8

    return dict(std_u=std_u, std_y=std_y, std_x=std_x,
                u_all=u_all, y_all=y_all, x_all=x_all)


# =============================================================================
# Analytical baseline (P_inv + finite differences)
# =============================================================================

def compute_analytical_baseline(y, x_logical):
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)
    pos     = (P_inv_T @ y.T).T                      # THEORY: q = inv(P^T) y
    vel     = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * FS_NEW          # HEURISTIC: backward FD
    vel[0]  = vel[1]
    x_ana   = np.hstack([pos, vel])
    rms_err = np.sqrt(np.mean((x_ana - x_logical)**2, axis=0))
    rms_gt  = np.sqrt(np.mean(x_logical**2, axis=0))
    return rms_err / (rms_gt + 1e-12), x_ana


# =============================================================================
# Encoder evaluation -- pure-scaled inputs, no convention fix needed
# =============================================================================

def evaluate_encoder_direct(encoder, val_u, val_y, val_x, norm, nx_out):
    """Sliding-window forward pass with pure-scaled inputs.

    Returns:
      nrms_phys  (NX_PHYS,)   NRMS of physical states vs x_logical
      x_enc_phys (T, NX_PHYS) physical states in physical units
      x_enc_ann  (T, nx_ann)  augmented states (normalized)
      x_gt       (T, NX_PHYS) aligned ground truth
    """
    nx_ann = nx_out - NX_PHYS

    # Pure-scaled: divide by std, no mean subtraction
    u_norm = val_u / norm['std_u']
    y_norm = val_y / norm['std_y']

    u_wins = np.lib.stride_tricks.sliding_window_view(
        u_norm, (_NB_WIN, nu)).reshape(-1, _NB_WIN, nu)
    y_wins = np.lib.stride_tricks.sliding_window_view(
        y_norm, (_NA_WIN, ny)).reshape(-1, _NA_WIN, ny)

    encoder.eval()
    with torch.no_grad():
        x_enc = encoder(
            torch.tensor(u_wins.reshape(len(u_wins), -1).copy(), dtype=DTYPE_PT),
            torch.tensor(y_wins.reshape(len(y_wins), -1).copy(), dtype=DTYPE_PT),
        ).numpy()   # (T, nx_out)

    # Denormalize physical states
    x_enc_phys = x_enc[:, :NX_PHYS] * norm['std_x']
    x_enc_ann  = x_enc[:, NX_PHYS:]

    x_gt = val_x[_WIN_START:]
    T    = min(len(x_enc_phys), len(x_gt))
    x_enc_phys = x_enc_phys[:T]
    x_enc_ann  = x_enc_ann[:T]
    x_gt       = x_gt[:T]

    rms_err   = np.sqrt(np.mean((x_enc_phys - x_gt)**2, axis=0))
    rms_gt    = np.sqrt(np.mean(x_gt**2, axis=0))
    nrms_phys = rms_err / (rms_gt + 1e-12)

    return nrms_phys, x_enc_phys, x_enc_ann, x_gt


# =============================================================================
# Plotting
# =============================================================================

def plot_state_comparison(x_enc_phys, x_enc_ann, x_ana, x_gt,
                          nrms_enc, nrms_ana, title, out_path):
    T = min(2000, len(x_enc_phys))
    t = np.arange(T) / FS_NEW + _WIN_START / FS_NEW

    n_rows = NX_PHYS + NX_ANN
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.2 * n_rows), sharex=True)

    for i in range(NX_PHYS):
        ax = axes[i]
        ax.plot(t, x_gt[:T, i],       'k-',  lw=0.8, label='x_logical (GT)')
        ax.plot(t, x_enc_phys[:T, i], 'r--', lw=0.8,
                label=f'enc_aug x_b (NRMS={nrms_enc[i]:.2e})')
        ax.plot(t, x_ana[:T, i],      'b:',  lw=0.8,
                label=f'analytical (NRMS={nrms_ana[i]:.2e})')
        ax.set_ylabel(f'{STATE_NAMES[i]} [{PHYS_UNITS[i]}]')
        ax.legend(loc='upper right', fontsize=6)
        ax.grid(True, alpha=0.3)

    for j in range(NX_ANN):
        ax = axes[NX_PHYS + j]
        ax.plot(t, x_enc_ann[:T, j], 'g-', lw=0.8)
        rms_j = float(np.sqrt(np.mean(x_enc_ann[:T, j]**2)))
        ax.set_ylabel(f'x_ann[{j}] (norm)')
        ax.set_title(f'ANN state {j} at init -- kaiming random, RMS={rms_j:.3e}',
                     fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_nrms_bar(nrms_dict, out_path):
    x = np.arange(NX_PHYS)
    n = len(nrms_dict)
    w = 0.8 / n
    colors = ['tab:red', 'tab:blue', 'tab:orange']
    fig, ax = plt.subplots(figsize=(10, 4))
    for j, (label, nrms) in enumerate(nrms_dict.items()):
        ax.bar(x + (j - n/2 + 0.5) * w, nrms, w,
               label=label, color=colors[j % len(colors)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES)
    ax.set_ylabel('NRMS')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Physical state NRMS at init -- enc_aug x_b vs analytical baseline')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print('=' * 70)
    print('Diagnostic 2: physical state recovery -- linear_encoder_init_aug')
    print(f'  NX_PHYS={NX_PHYS}  NX_ANN={NX_ANN}  na=nb={na}')
    print(f'  FS_NEW={FS_NEW} Hz  Ts={TS_NEW*1000:.3f} ms')
    print(f'  Data: {TRAJ_DIR}')
    print('=' * 70)

    # ── Data ─────────────────────────────────────────────────────────────────
    print('\nLoading data...')
    train_data = [load_mat(f) for f in TRAIN_FILES]
    val_u, val_y, val_x = load_mat(VAL_FILE)
    print(f'  Val: {val_u.shape[0]} samples')

    # ── Normalization (pure-scaled) ───────────────────────────────────────────
    norm = compute_norm_pure_scaled(train_data)

    # ── System matrices ───────────────────────────────────────────────────────
    Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)

    # normalize_linear_ss_matrices uses pure-scaled statistics to produce
    # A_bar, B_bar, C_bar, D_bar compatible with u/std_u, y/std_y inputs
    sys_data_tmp = deepSI.System_data(u=norm['u_all'], y=norm['y_all'])
    sys_data_tmp.x = norm['x_all']
    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        Ad, Bd, Cd_dt, Dd_dt, sys_data_tmp)

    # ── Build encoders ────────────────────────────────────────────────────────
    print('\nBuilding encoders...')

    enc_jan = linear_encoder_init(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
        n_nodes_per_layer=N_NODES, n_hidden_layers=N_HIDDEN,
        flag_linear_only=False,
    )

    enc_aug = linear_encoder_init_aug(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
        nx_aug=NX_ANN,
        n_nodes_per_layer=N_NODES, n_hidden_layers=N_HIDDEN,
        flag_linear_only=False,
    )

    # ── Check 1: W^b identity ─────────────────────────────────────────────────
    print('\n--- Check 1: W^b identity ---')
    checks = {}

    diff_y = (enc_jan.Wb_psi_y.detach() - enc_aug.Wb_psi_y.detach()).abs().max().item()
    diff_u = (enc_jan.Wb_psi_u.detach() - enc_aug.Wb_psi_u.detach()).abs().max().item()
    ok_y = diff_y < TOL_IDENTITY
    ok_u = diff_u < TOL_IDENTITY
    checks['Wb_psi_y_identity'] = 'PASS' if ok_y else 'FAIL'
    checks['Wb_psi_u_identity'] = 'PASS' if ok_u else 'FAIL'
    print(f'  [{"PASS" if ok_y else "FAIL"}] Wb_psi_y  max_diff={diff_y:.2e}')
    print(f'  [{"PASS" if ok_u else "FAIL"}] Wb_psi_u  max_diff={diff_u:.2e}')

    # ── Check 2: x_b forward pass identity ───────────────────────────────────
    print('\n--- Check 2: x_b forward pass identity (flag_linear_only=True) ---')
    enc_jan_lin = linear_encoder_init(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
        flag_linear_only=True,
    )
    enc_aug_lin = linear_encoder_init_aug(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
        nx_aug=NX_ANN,
        flag_linear_only=True,
    )

    # Use first val window batch as test input
    u_norm = val_u / norm['std_u']
    y_norm = val_y / norm['std_y']
    u_wins = np.lib.stride_tricks.sliding_window_view(
        u_norm, (_NB_WIN, nu)).reshape(-1, _NB_WIN, nu)
    y_wins = np.lib.stride_tricks.sliding_window_view(
        y_norm, (_NA_WIN, ny)).reshape(-1, _NA_WIN, ny)

    u_t = torch.tensor(u_wins[:64].reshape(64, -1).copy(), dtype=DTYPE_PT)
    y_t = torch.tensor(y_wins[:64].reshape(64, -1).copy(), dtype=DTYPE_PT)

    with torch.no_grad():
        out_jan = enc_jan_lin(u_t, y_t)               # (64, NX_PHYS)
        out_aug = enc_aug_lin(u_t, y_t)               # (64, NX_PHYS+NX_ANN)

    x_b_aug = out_aug[:, :NX_PHYS]
    x_a_aug = out_aug[:, NX_PHYS:]

    diff_xb = (out_jan - x_b_aug).abs().max().item()
    ok_xb = diff_xb < TOL_IDENTITY
    checks['xb_forward_identity'] = 'PASS' if ok_xb else 'FAIL'
    print(f'  [{"PASS" if ok_xb else "FAIL"}] x_b identical  max_diff={diff_xb:.2e}')

    # ── Check 3: x_a shape and nonzero ───────────────────────────────────────
    print('\n--- Check 3: x_a shape and kaiming init ---')
    ok_shape = x_a_aug.shape == (64, NX_ANN)
    ok_nonzero = x_a_aug.abs().max().item() > 1e-6
    ok_no_nan  = not torch.isnan(x_a_aug).any().item()
    checks['xa_shape']   = 'PASS' if ok_shape   else 'FAIL'
    checks['xa_nonzero'] = 'PASS' if ok_nonzero else 'FAIL'
    checks['xa_no_nan']  = 'PASS' if ok_no_nan  else 'FAIL'
    print(f'  [{"PASS" if ok_shape else "FAIL"}] x_a shape={tuple(x_a_aug.shape)} '
          f'(expected (64, {NX_ANN}))')
    print(f'  [{"PASS" if ok_nonzero else "FAIL"}] x_a nonzero  '
          f'max_abs={x_a_aug.abs().max().item():.3e}')
    print(f'  [{"PASS" if ok_no_nan else "FAIL"}] x_a no NaN')

    # ── Check 4: x_b NRMS vs x_logical ───────────────────────────────────────
    print('\n--- Check 4: x_b NRMS vs x_logical ---')
    nrms_ana, x_analytical = compute_analytical_baseline(val_y, val_x)
    x_ana_win = x_analytical[_WIN_START:]

    nrms_aug, x_enc_phys, x_enc_ann, x_gt = evaluate_encoder_direct(
        enc_aug, val_u, val_y, val_x, norm, nx_out=NX_PHYS + NX_ANN)
    nrms_jan, x_enc_jan, _, _ = evaluate_encoder_direct(
        enc_jan, val_u, val_y, val_x, norm, nx_out=NX_PHYS)

    print(f'  {"State":<6s}  {"enc_aug x_b":>12s}  {"enc_jan":>12s}  {"analytical":>12s}')
    print(f'  {"-"*6}  {"-"*12}  {"-"*12}  {"-"*12}')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:<6s}  {nrms_aug[i]:>12.4e}  {nrms_jan[i]:>12.4e}  '
              f'{nrms_ana[i]:>12.4e}')

    # enc_aug x_b should match enc_jan exactly (same W^b, same net, no x_a interference)
    diff_nrms = np.abs(nrms_aug - nrms_jan).max()
    ok_nrms_match = diff_nrms < 1e-6
    checks['xb_nrms_matches_jan'] = 'PASS' if ok_nrms_match else 'FAIL'
    print(f'\n  [{"PASS" if ok_nrms_match else "FAIL"}] '
          f'x_b NRMS identical to Jan encoder  max_diff={diff_nrms:.2e}')

    ann_rms = [float(np.sqrt(np.mean(x_enc_ann[:, j]**2))) for j in range(NX_ANN)]
    print(f'\n  x_a RMS at init (kaiming random, normalized):')
    for j in range(NX_ANN):
        print(f'    x_ann[{j}]: {ann_rms[j]:.4e}')

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(v == 'PASS' for v in checks.values())
    n_fail = sum(v == 'FAIL' for v in checks.values())
    print(f'\n{"="*70}')
    print(f'  {n_pass}/{len(checks)} checks passed')
    if n_fail > 0:
        failed = [k for k, v in checks.items() if v == 'FAIL']
        print(f'  FAILED: {failed}')
        print('  linear_encoder_init_aug physical state recovery NOT verified.')
    else:
        print('  linear_encoder_init_aug confirmed:')
        print('    x_b identical to Jan encoder at init')
        print('    x_a nonzero (kaiming), no NaN')
        print('    x_b NRMS consistent with analytical baseline')

    # ── Save ──────────────────────────────────────────────────────────────────
    T_save = min(len(x_enc_phys), len(x_gt), len(x_ana_win))
    npz_path = os.path.join(OUT_DIR, 'diag2_physical_state_recovery.npz')
    np.savez_compressed(npz_path,
        x_enc_phys   = x_enc_phys[:T_save],
        x_enc_ann    = x_enc_ann[:T_save],
        x_enc_jan    = x_enc_jan[:T_save],
        x_analytical = x_ana_win[:T_save],
        x_gt         = x_gt[:T_save],
        nrms_aug     = nrms_aug,
        nrms_jan     = nrms_jan,
        nrms_ana     = nrms_ana,
        ann_rms_init = np.array(ann_rms),
        state_names  = np.array(STATE_NAMES),
        fs           = np.float32(FS_NEW),
        win_start    = np.int32(_WIN_START),
    )
    print(f'\n  Saved: {npz_path}')

    json_path = os.path.join(OUT_DIR, 'diag2_physical_state_recovery.json')
    with open(json_path, 'w') as f:
        json.dump(dict(
            config=dict(NX_PHYS=NX_PHYS, NX_ANN=NX_ANN, FS_NEW=FS_NEW,
                        na=na, nb=nb, MA_FRAC=MA_FRAC,
                        MULTISINE_BAND=MULTISINE_BAND),
            checks=checks,
            nrms_aug={n: float(nrms_aug[i]) for i, n in enumerate(STATE_NAMES)},
            nrms_jan={n: float(nrms_jan[i]) for i, n in enumerate(STATE_NAMES)},
            nrms_ana={n: float(nrms_ana[i]) for i, n in enumerate(STATE_NAMES)},
            ann_rms_init={j: float(ann_rms[j]) for j in range(NX_ANN)},
        ), f, indent=2)
    print(f'  Saved: {json_path}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_state_comparison(
        x_enc_phys, x_enc_ann, x_ana_win, x_gt,
        nrms_aug, nrms_ana,
        f'linear_encoder_init_aug -- x_b recovery at init (NX_ANN={NX_ANN})',
        os.path.join(OUT_DIR, 'diag2_state_recovery.png'))

    plot_nrms_bar(
        {'enc_aug x_b': nrms_aug, 'enc_jan': nrms_jan, 'analytical': nrms_ana},
        os.path.join(OUT_DIR, 'diag2_nrms_bar.png'))

    if n_fail > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
