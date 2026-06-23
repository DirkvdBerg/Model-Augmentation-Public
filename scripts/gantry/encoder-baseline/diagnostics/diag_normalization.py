"""
diag_normalization.py
---------------------
Diagnose why linear_encoder_init velocity NRMS is poor (~0.5) while
positions are near-perfect (~5e-5).

Root cause hypothesis: normalize_linear_ss_matrices uses pure scaling
(x/std_x, u/std_u, y/std_y) but the pipeline feeds mean-subtracted data
((x-x_mean)/std_x, (u-u_mean)/std_u, (y-y0)/std_y) to the encoder.

Checks:
  1. Output equation: C_bar @ (x/std_x) vs y/std_y (pure-scaled, should work)
  2. One-step prediction: both conventions
  3. O_n condition number
  4. Manual reconstruction: pure-scaled convention (should be near-perfect)
  5. Manual reconstruction: pipeline convention (should show the error)
  6. Bias offset: compute and verify the correction term
  7. Corrected reconstruction: with bias fix (should restore near-perfect)
  8. Std source mismatch: normalize_linear_ss_matrices vs pipeline std_x
  9. x_logical (MATLAB) vs finite-diff (Python pipeline) comparison
  10. Time ordering: does O_n convention match y_hist ordering?

Uses T3 (Y=0) where the LTI linearization is exact.

Usage:
    conda run -n GraduationProject python scripts/gantry/encoder-baseline/diagnostics/diag_normalization.py
"""

import os
import sys
import numpy as np
from scipy.io import loadmat

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import deepSI
from model_augmentation.utils.utils import normalize_linear_ss_matrices
from model_augmentation.systems.gantry_ss import Cd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize

# =============================================================================
# Config
# =============================================================================
NX_PHYS = 6
nu = 3
ny = 3
FS_ORIG = 20000
FS_NEW = 4000
D = FS_ORIG // FS_NEW
TS_NEW = 1.0 / FS_NEW
na = 25  # HEURISTIC: Jan's 4*NX_PHYS+1
nb = 25
DTYPE = np.float64  # float64 for diagnostics — rules out precision issues

TRAJ_DIR = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab', 'multisine', 'baseline')
TRAIN_FILES = [
    'T1_Y_sweep_conservative.mat', 'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat', 'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat', 'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat', 'T8_X_sym_anti_Y_sweep.mat',
]
DIAG_FILE = 'T3_X_sym_Y000.mat'  # Y=0 → LTI linearization is exact
STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']

BASELINE_NPZ = os.path.join(
    PROJECT_ROOT, 'data', 'gantry', 'baseline_simulations',
    'multisine_LPV', 'baseline_states.npz')


def load_mat(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = d['u_total'][::D].astype(DTYPE) if 'u_total' in d else d['u'][::D].astype(DTYPE)
    y = d['y'][::D].astype(DTYPE)
    x = d['x_logical'][::D].astype(DTYPE)
    return u, y, x


def nrms(x_hat, x_true):
    """Per-channel NRMS."""
    return np.sqrt(np.mean((x_hat - x_true)**2, axis=0)) / (
        np.sqrt(np.mean(x_true**2, axis=0)) + 1e-12)


def print_state_table(header, cols, data_cols):
    """Print a table with state rows and named columns."""
    col_w = max(16, max(len(h) for h in cols))
    print(f'  {"State":<6s}  ' + '  '.join(f'{h:>{col_w}s}' for h in cols))
    print(f'  {"-"*6}  ' + '  '.join(f'{"-"*col_w}' for _ in cols))
    for i, name in enumerate(STATE_NAMES):
        vals = '  '.join(f'{d[i]:>{col_w}.4e}' for d in data_cols)
        print(f'  {name:<6s}  {vals}')


def main():
    print('=' * 70)
    print('Normalization diagnostic')
    print('=' * 70)

    # =========================================================================
    # Load data and compute normalization (replicating gantry_interconnect_dynamic.py)
    # =========================================================================
    train_data = [load_mat(f) for f in TRAIN_FILES]
    u_all = np.concatenate([u for u, _, _ in train_data])
    y_all = np.concatenate([y for _, y, _ in train_data])
    x_logical_all = np.concatenate([x for _, _, x in train_data])

    # Pipeline normalization constants (lines 148-153 of gantry_interconnect_dynamic.py)
    x_mean = x_logical_all.mean(axis=0)
    std_x = x_logical_all.std(axis=0) + 1e-8
    u_mean = u_all.mean(axis=0)
    std_u = u_all.std(axis=0) + 1e-8
    ystd = y_all.std(axis=0) + 1e-8
    y0 = Cd.numpy().astype(DTYPE) @ x_mean  # Cd @ x_mean

    print('\n--- Pipeline normalization constants ---')
    print(f'  x_mean = {x_mean}')
    print(f'  std_x  = {std_x}')
    print(f'  u_mean = {u_mean}')
    print(f'  std_u  = {std_u}')
    print(f'  ystd   = {ystd}')
    print(f'  y0     = {y0}  (Cd @ x_mean)')
    print(f'  y_mean = {y_all.mean(axis=0)}  (empirical)')
    print(f'  y0 - y_mean = {y0 - y_all.mean(axis=0)}')

    # --- DT system matrices ---
    Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)
    Ad = Ad.astype(DTYPE)
    Bd = Bd.astype(DTYPE)
    Cd_dt = Cd_dt.astype(DTYPE)
    Dd_dt = Dd_dt.astype(DTYPE)

    print(f'\n--- DT system at Y_op=0, dt={TS_NEW} ---')
    print(f'  Ad eigval magnitudes: {np.sort(np.abs(np.linalg.eigvals(Ad)))[::-1]}')

    # =========================================================================
    # CHECK 8: Std source mismatch
    # =========================================================================
    print('\n' + '=' * 70)
    print('CHECK 8: Std source — normalize_linear_ss_matrices vs pipeline')
    print('=' * 70)

    # What normalize_linear_ss_matrices uses: std from sys_data.x
    # In gantry_interconnect_dynamic.py line 242-245, sys_data.x comes from either
    # baseline_states.npz or finite-diff (x_all fallback).
    # The pipeline's std_x (line 149) comes from x_all (finite-diff of P_inv @ y).
    # But step0_init_diagnostic.py uses x_logical from .mat files!

    # Compute finite-diff states (what the pipeline actually uses)
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE)
    x_fd_list = []
    for t_u, t_y, _ in train_data:
        pos = (P_inv_T @ t_y.T).T
        vel = np.diff(pos, axis=0) * FS_NEW
        vel = np.vstack([vel[:1], vel])
        x_fd_list.append(np.hstack([pos, vel]))
    x_fd_all = np.concatenate(x_fd_list)

    std_x_fd = x_fd_all.std(axis=0) + 1e-8
    std_x_xlog = x_logical_all.std(axis=0) + 1e-8

    # Check baseline_states.npz
    if os.path.exists(BASELINE_NPZ):
        bl = np.load(BASELINE_NPZ, allow_pickle=True)
        x_bl_all = np.concatenate(bl['x_train_phys'])
        std_x_bl = x_bl_all.std(axis=0) + 1e-8
        print(f'  baseline_states.npz found ({x_bl_all.shape[0]} samples)')
    else:
        std_x_bl = std_x_fd
        print(f'  baseline_states.npz NOT found, would fall back to finite-diff')

    print(f'\n  std_x sources:')
    print(f'  {"State":<6s}  {"x_logical (.mat)":>18s}  {"finite-diff (pipe)":>18s}  {"baseline_sim":>18s}  {"ratio xlog/fd":>14s}')
    print(f'  {"-"*6}  {"-"*18}  {"-"*18}  {"-"*18}  {"-"*14}')
    for i, name in enumerate(STATE_NAMES):
        ratio = std_x_xlog[i] / std_x_fd[i]
        print(f'  {name:<6s}  {std_x_xlog[i]:>18.6e}  {std_x_fd[i]:>18.6e}  {std_x_bl[i]:>18.6e}  {ratio:>14.6f}')

    if np.max(np.abs(std_x_xlog / std_x_fd - 1)) > 0.01:
        print('\n  WARNING: >1% std mismatch between x_logical and finite-diff!')
        print('  step0 uses x_logical for normalization but pipeline may use finite-diff.')
    if np.max(np.abs(std_x_bl / std_x_fd - 1)) > 0.01:
        print('\n  WARNING: >1% std mismatch between baseline_sim and finite-diff!')
        print('  normalize_linear_ss_matrices would use different scaling than pipeline.')

    # =========================================================================
    # CHECK 9: x_logical (MATLAB) vs finite-diff (Python pipeline)
    # =========================================================================
    print('\n' + '=' * 70)
    print('CHECK 9: x_logical (MATLAB) vs finite-diff (Python pipeline)')
    print('=' * 70)

    # Use T3 for comparison
    u_t3, y_t3, x_t3 = load_mat(DIAG_FILE)
    pos_t3_fd = (P_inv_T @ y_t3.T).T
    vel_t3_fd = np.diff(pos_t3_fd, axis=0) * FS_NEW
    vel_t3_fd = np.vstack([vel_t3_fd[:1], vel_t3_fd])
    x_t3_fd = np.hstack([pos_t3_fd, vel_t3_fd])

    diff_xlog_fd = nrms(x_t3_fd, x_t3)
    print(f'  {DIAG_FILE}: NRMS(finite-diff, x_logical):')
    for i, name in enumerate(STATE_NAMES):
        print(f'    {name}: {diff_xlog_fd[i]:.4e}')
    print(f'  Positions should be identical (same P_inv @ y).')
    print(f'  Velocities differ: MATLAB gradient() (central diff) vs np.diff (forward diff).')

    # =========================================================================
    # Normalize matrices using the SAME data as step0 (x_logical)
    # =========================================================================
    sys_data = deepSI.System_data(u=u_all, y=y_all)
    sys_data.x = x_logical_all
    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        Ad, Bd, Cd_dt, Dd_dt, sys_data)

    # Also normalize using what the pipeline actually passes (baseline_sim or fd)
    sys_data_pipe = deepSI.System_data(u=u_all, y=y_all)
    if os.path.exists(BASELINE_NPZ):
        sys_data_pipe.x = x_bl_all
    else:
        sys_data_pipe.x = x_fd_all
    Ad_bar_pipe, Bd_bar_pipe, Cd_bar_pipe, Dd_bar_pipe = normalize_linear_ss_matrices(
        Ad, Bd, Cd_dt, Dd_dt, sys_data_pipe)

    print(f'\n  Max |Ad_bar - Ad_bar_pipe| = {np.max(np.abs(Ad_bar - Ad_bar_pipe)):.4e}')
    print(f'  Max |Bd_bar - Bd_bar_pipe| = {np.max(np.abs(Bd_bar - Bd_bar_pipe)):.4e}')
    if np.max(np.abs(Ad_bar - Ad_bar_pipe)) > 1e-3:
        print('  WARNING: Matrix normalization differs between x_logical and pipeline x!')

    # =========================================================================
    # Load T3 diagnostic trajectory
    # =========================================================================
    N = u_t3.shape[0]
    print(f'\n--- Diagnostic trajectory: {DIAG_FILE}, N={N} ---')

    # Pre-compute normalized signals in both conventions
    x_scaled = x_t3 / std_x                        # pure-scaled (what matrices expect)
    u_scaled = u_t3 / std_u
    y_scaled = y_t3 / ystd

    x_norm = (x_t3 - x_mean) / std_x               # pipeline convention
    u_norm = (u_t3 - u_mean) / std_u
    y_norm = (y_t3 - y0) / ystd

    # =========================================================================
    # CHECK 1: Output equation C_bar @ x ≈ y
    # =========================================================================
    print('\n' + '=' * 70)
    print('CHECK 1: Output equation C_bar @ x ≈ y')
    print('=' * 70)

    # Pure-scaled: C_bar @ (x/std_x) should equal y/std_y
    y_pred_scaled = (Cd_bar @ x_scaled.T).T
    err_scaled = np.sqrt(np.mean((y_pred_scaled - y_scaled)**2, axis=0))

    # Pipeline: C_bar @ ((x-x_mean)/std_x) vs (y-y0)/std_y
    y_pred_norm = (Cd_bar @ x_norm.T).T
    err_norm = np.sqrt(np.mean((y_pred_norm - y_norm)**2, axis=0))

    print(f'  Pure-scaled: C_bar @ (x/std_x) vs y/std_y')
    print(f'    RMS err per output: {err_scaled}')
    print(f'  Pipeline:    C_bar @ x_norm vs y_norm')
    print(f'    RMS err per output: {err_norm}')

    # Explain the offset
    # C_bar @ x_scaled = y_scaled  (true)
    # C_bar @ x_norm = C_bar @ (x_scaled - x_mean/std_x)
    #                = y_scaled - C_bar @ (x_mean/std_x)
    # y_norm = y_scaled - y0/ystd
    # So: C_bar @ x_norm - y_norm = y0/ystd - C_bar @ (x_mean/std_x)
    offset_y = y0 / ystd - (Cd_bar @ (x_mean / std_x))
    print(f'\n  Output offset = y0/ystd - C_bar @ (x_mean/std_x):')
    print(f'    {offset_y}')
    print(f'    (should be ~0 if y0 = Cd @ x_mean and C_bar = Ty @ Cd @ Tix)')
    # Indeed: y0/ystd = Cd@x_mean/ystd, C_bar@(x_mean/std_x) = Ty@Cd@Tix@(x_mean/std_x)
    #       = (1/ystd)*Cd*(std_x)*(x_mean/std_x) = Cd@x_mean/ystd  → exact match!

    # =========================================================================
    # CHECK 2: One-step prediction A_bar @ x + B_bar @ u ≈ x_next
    # =========================================================================
    print('\n' + '=' * 70)
    print('CHECK 2: One-step prediction')
    print('=' * 70)

    # Pure-scaled
    x_pred_scaled = (Ad_bar @ x_scaled[:-1].T + Bd_bar @ u_scaled[:-1].T).T
    err_1s_scaled = np.sqrt(np.mean((x_pred_scaled - x_scaled[1:])**2, axis=0))

    # Pipeline convention
    x_pred_norm = (Ad_bar @ x_norm[:-1].T + Bd_bar @ u_norm[:-1].T).T
    err_1s_norm = np.sqrt(np.mean((x_pred_norm - x_norm[1:])**2, axis=0))

    print(f'  Pure-scaled (no mean sub):')
    print_state_table('', ['RMS err'], [err_1s_scaled])
    print(f'\n  Pipeline (mean-subtracted):')
    print_state_table('', ['RMS err'], [err_1s_norm])

    # The offset for one-step:
    # True:      x_scaled[k+1] = Ad_bar @ x_scaled[k] + Bd_bar @ u_scaled[k]
    # Pipeline:  Ad_bar @ x_norm[k] + Bd_bar @ u_norm[k]
    #          = Ad_bar @ (x_scaled[k] - x_mean/std_x) + Bd_bar @ (u_scaled[k] - u_mean/std_u)
    #          = x_scaled[k+1] - Ad_bar @ (x_mean/std_x) - Bd_bar @ (u_mean/std_u)
    # Should be: x_norm[k+1] = x_scaled[k+1] - x_mean/std_x
    # Offset:  x_mean/std_x - Ad_bar @ (x_mean/std_x) - Bd_bar @ (u_mean/std_u)
    #        = (I - Ad_bar) @ (x_mean/std_x) - Bd_bar @ (u_mean/std_u)
    offset_1s = (np.eye(NX_PHYS) - Ad_bar) @ (x_mean / std_x) - Bd_bar @ (u_mean / std_u)
    print(f'\n  One-step offset = (I - Ad_bar) @ (x_mean/std_x) - Bd_bar @ (u_mean/std_u):')
    for i, name in enumerate(STATE_NAMES):
        print(f'    {name}: {offset_1s[i]:.6e}')

    # =========================================================================
    # CHECK 3: Observability matrix conditioning
    # =========================================================================
    print('\n' + '=' * 70)
    print('CHECK 3: Observability matrix O_n')
    print('=' * 70)

    n = na
    O_n = np.zeros(((n + 1) * ny, NX_PHYS))
    for i in range(n + 1):
        O_n[i * ny: (i + 1) * ny, :] = Cd_bar @ np.linalg.matrix_power(Ad_bar, i)

    svs = np.linalg.svd(O_n, compute_uv=False)
    print(f'  O_n shape: {O_n.shape}')
    print(f'  Rank: {np.linalg.matrix_rank(O_n)} (need {NX_PHYS})')
    print(f'  Condition: {np.linalg.cond(O_n):.4e}')
    print(f'  Top {NX_PHYS} singular values: {svs[:NX_PHYS]}')
    print(f'  SV ratio max/min: {svs[0]/svs[NX_PHYS-1]:.4e}')

    # =========================================================================
    # Build Wb_psi_y, Wb_psi_u (replicating linear_encoder_init exactly)
    # =========================================================================
    O_inv = np.linalg.pinv(O_n)
    A_n = np.linalg.matrix_power(Ad_bar, n)

    # gamma_n: x[k+n] = A^n @ x[k] + sum_{i=0}^{n-1} A^{n-i-1} B u[k+i]
    gamma_n = np.zeros((NX_PHYS, (n + 1) * nu))
    for i in range(n):
        gamma_n[:, i * nu: (i + 1) * nu] = np.linalg.matrix_power(Ad_bar, n - i - 1) @ Bd_bar

    # Gamma_n with FLIPPED indices (same as linear_encoder_init lines 228-243)
    Gamma_n = np.zeros(((n + 1) * ny, (n + 1) * nu))
    for i in range(n + 1):
        for j in range(i, n + 1):
            fi = n - i  # flipped row
            fj = n - j  # flipped col
            if i != j:
                Gamma_n[fi * ny:(fi + 1) * ny, fj * nu:(fj + 1) * nu] = \
                    Cd_bar @ np.linalg.matrix_power(Ad_bar, j - i - 1) @ Bd_bar
            else:
                Gamma_n[fi * ny:(fi + 1) * ny, fj * nu:(fj + 1) * nu] = Dd_bar

    Wb_psi_y = A_n @ O_inv           # (6, 78)
    Wb_psi_u = -A_n @ O_inv @ Gamma_n + gamma_n  # (6, 78)

    # =========================================================================
    # CHECK 4: Manual reconstruction — pure-scaled (should be near-perfect)
    # =========================================================================
    print('\n' + '=' * 70)
    print('CHECK 4: Reconstruction with PURE-SCALED data (no mean sub)')
    print('=' * 70)

    hist_len = n + 1  # 26 samples
    M = N - hist_len
    x_hat_scaled = np.zeros((M, NX_PHYS))

    for idx in range(M):
        k = hist_len + idx - 1  # target sample (encoder maps to x[k])
        yh = (y_t3[k - n: k + 1] / ystd).flatten()  # (78,)
        uh = (u_t3[k - n: k + 1] / std_u).flatten()  # (78,)
        x_hat_scaled[idx] = Wb_psi_y @ yh + Wb_psi_u @ uh

    x_true_scaled = x_t3[hist_len - 1: hist_len - 1 + M] / std_x
    nrms_scaled = nrms(x_hat_scaled, x_true_scaled)

    print(f'  Batch NRMS (pure-scaled, {M} windows):')
    print_state_table('', ['NRMS'], [nrms_scaled])

    if np.max(nrms_scaled) < 1e-6:
        print(f'\n  PASS: Pure-scaled reconstruction is near-perfect ({np.max(nrms_scaled):.2e})')
    else:
        print(f'\n  FAIL: Pure-scaled reconstruction has errors > 1e-6!')
        print(f'  This suggests a problem with the matrices or O_n, not just mean subtraction.')

    # =========================================================================
    # CHECK 5: Manual reconstruction — pipeline convention (mean-subtracted)
    # =========================================================================
    print('\n' + '=' * 70)
    print('CHECK 5: Reconstruction with PIPELINE data (mean-subtracted)')
    print('=' * 70)
    print('  This is what the encoder actually receives at inference.')

    x_hat_pipe = np.zeros((M, NX_PHYS))
    for idx in range(M):
        k = hist_len + idx - 1
        yh = ((y_t3[k - n: k + 1] - y0) / ystd).flatten()
        uh = ((u_t3[k - n: k + 1] - u_mean) / std_u).flatten()
        x_hat_pipe[idx] = Wb_psi_y @ yh + Wb_psi_u @ uh

    x_true_norm = (x_t3[hist_len - 1: hist_len - 1 + M] - x_mean) / std_x
    nrms_pipe = nrms(x_hat_pipe, x_true_norm)

    print(f'  Batch NRMS (pipeline, {M} windows):')
    print_state_table('', ['NRMS (pipeline)', 'NRMS (pure-scaled)'],
                      [nrms_pipe, nrms_scaled])

    if np.max(nrms_pipe) > 10 * np.max(nrms_scaled):
        print(f'\n  CONFIRMED: Mean subtraction causes {np.max(nrms_pipe)/np.max(nrms_scaled):.0f}x '
              f'worse NRMS.')
    else:
        print(f'\n  Mean subtraction does NOT explain the error.')

    # =========================================================================
    # CHECK 6: Bias offset computation
    # =========================================================================
    print('\n' + '=' * 70)
    print('CHECK 6: Bias offset from mean subtraction')
    print('=' * 70)

    # The encoder computes: x_hat = Wb_psi_y @ y_hist_pipe + Wb_psi_u @ u_hist_pipe
    # But it should compute: Wb_psi_y @ y_hist_scaled + Wb_psi_u @ u_hist_scaled
    #
    # Difference: - Wb_psi_y @ [y0/ystd repeated] - Wb_psi_u @ [u_mean/std_u repeated]
    #
    # The encoder output should be x_norm = (x - x_mean)/std_x = x_scaled - x_mean/std_x
    # So the correct output is: x_scaled - x_mean/std_x
    # The actual output is:     x_scaled - Wb_psi_y @ y0_rep - Wb_psi_u @ u_mean_rep
    #
    # Bias correction: x_mean/std_x - Wb_psi_y @ y0_rep - Wb_psi_u @ u_mean_rep
    #   (add this to encoder output to fix it)

    # Build repeated mean vectors (n+1 copies of the per-channel mean)
    y0_rep = np.tile(y0 / ystd, n + 1)           # ((n+1)*ny,)
    u_mean_rep = np.tile(u_mean / std_u, n + 1)  # ((n+1)*nu,)

    bias_correction = x_mean / std_x - Wb_psi_y @ y0_rep - Wb_psi_u @ u_mean_rep

    print(f'  Bias correction per state:')
    for i, name in enumerate(STATE_NAMES):
        print(f'    {name}: {bias_correction[i]:>12.6f}')

    # Verify: the empirical offset should match the theoretical bias
    empirical_offset = (x_hat_pipe - x_true_norm).mean(axis=0)
    expected_offset = -bias_correction  # the error is the negative of the correction

    print(f'\n  Verification:')
    print(f'  {"State":<6s}  {"empirical offset":>18s}  {"theoretical":>18s}  {"match?":>10s}')
    print(f'  {"-"*6}  {"-"*18}  {"-"*18}  {"-"*10}')
    for i, name in enumerate(STATE_NAMES):
        match = abs(empirical_offset[i] - expected_offset[i]) < 1e-4
        print(f'  {name:<6s}  {empirical_offset[i]:>18.6e}  {expected_offset[i]:>18.6e}  '
              f'{"YES" if match else "NO":>10s}')

    # =========================================================================
    # CHECK 7: Corrected reconstruction (with bias fix)
    # =========================================================================
    print('\n' + '=' * 70)
    print('CHECK 7: Corrected reconstruction (pipeline + bias correction)')
    print('=' * 70)

    x_hat_corrected = x_hat_pipe + bias_correction[np.newaxis, :]
    nrms_corrected = nrms(x_hat_corrected, x_true_norm)

    print(f'  Batch NRMS comparison:')
    print_state_table('', ['uncorrected', 'corrected', 'pure-scaled'],
                      [nrms_pipe, nrms_corrected, nrms_scaled])

    if np.max(nrms_corrected) < 1e-6:
        print(f'\n  PASS: Bias correction restores near-perfect reconstruction.')
        print(f'  FIX: Add bias_correction as a constant bias in the encoder.')
    else:
        print(f'\n  Bias correction helps but does NOT fully fix the issue.')
        print(f'  There may be additional problems (time ordering, std mismatch).')

    # =========================================================================
    # CHECK 10: Time ordering
    # =========================================================================
    print('\n' + '=' * 70)
    print('CHECK 10: Time ordering analysis')
    print('=' * 70)

    # O_n convention: row 0 = C @ A^0 = C (most recent observation)
    #                 row n = C @ A^n (oldest observation)
    # y_hist convention (from to_hist_future_data): y[k-n], ..., y[k]
    # In flat form: [y[k-n,0], y[k-n,1], y[k-n,2], y[k-n+1,0], ...]
    #
    # Wb_psi_y = A^n @ O_inv
    # O_n @ x[k] = [C @ x[k], C @ A @ x[k], ..., C @ A^n @ x[k]]
    #            = [y[k],      y[k-1],        ..., y[k-n]]  (newest to oldest)
    # But y_hist is [y[k-n], ..., y[k]] (oldest to newest) — OPPOSITE ORDER!
    #
    # The flipped indices in Gamma_n may be Jan's way of handling this.
    # Let's test by building a reversed O_n.

    print(f'  O_n convention: row 0 = C@A^0 (newest), row n = C@A^n (oldest)')
    print(f'  y_hist: [y[k-n], ..., y[k]] (oldest to newest)')
    print(f'  Gamma_n uses flipped indices: flipped_i = n - i')

    # Build reversed O_n (oldest first, matching y_hist order)
    O_n_rev = np.zeros(((n + 1) * ny, NX_PHYS))
    for i in range(n + 1):
        O_n_rev[i * ny: (i + 1) * ny, :] = Cd_bar @ np.linalg.matrix_power(Ad_bar, n - i)

    O_inv_rev = np.linalg.pinv(O_n_rev)
    Wb_psi_y_rev = A_n @ O_inv_rev

    # gamma_n reversed (oldest input first: u[k-n], u[k-n+1], ..., u[k])
    gamma_n_rev = np.zeros((NX_PHYS, (n + 1) * nu))
    for i in range(n):
        gamma_n_rev[:, i * nu: (i + 1) * nu] = np.linalg.matrix_power(Ad_bar, i) @ Bd_bar

    # Gamma_n reversed (no flipped indices, natural order)
    Gamma_n_rev = np.zeros(((n + 1) * ny, (n + 1) * nu))
    for i in range(n + 1):
        for j in range(i, n + 1):
            if i != j:
                Gamma_n_rev[i * ny:(i + 1) * ny, j * nu:(j + 1) * nu] = \
                    Cd_bar @ np.linalg.matrix_power(Ad_bar, j - i - 1) @ Bd_bar
            else:
                Gamma_n_rev[i * ny:(i + 1) * ny, j * nu:(j + 1) * nu] = Dd_bar

    Wb_psi_u_rev = -A_n @ O_inv_rev @ Gamma_n_rev + gamma_n_rev

    # Test reversed (pure-scaled)
    x_hat_rev_scaled = np.zeros((M, NX_PHYS))
    for idx in range(M):
        k = hist_len + idx - 1
        yh = (y_t3[k - n: k + 1] / ystd).flatten()
        uh = (u_t3[k - n: k + 1] / std_u).flatten()
        x_hat_rev_scaled[idx] = Wb_psi_y_rev @ yh + Wb_psi_u_rev @ uh

    nrms_rev = nrms(x_hat_rev_scaled, x_true_scaled)

    print(f'\n  Batch NRMS (pure-scaled):')
    print_state_table('', ['original O_n', 'reversed O_n'],
                      [nrms_scaled, nrms_rev])

    if np.max(nrms_rev) < np.max(nrms_scaled):
        print(f'\n  Reversed O_n is BETTER → time ordering mismatch!')
    elif np.max(nrms_scaled) < 1e-6 and np.max(nrms_rev) < 1e-6:
        print(f'\n  Both near-perfect. Gamma_n flipped indices compensate correctly.')
        print(f'  Time ordering is NOT the issue.')
    else:
        print(f'\n  Original O_n is better or equal.')

    # =========================================================================
    # CHECK 11: Wrapper fix — undo mean sub before Wb, subtract x_offset after
    # =========================================================================
    print('\n' + '=' * 70)
    print('CHECK 11: Wrapper convention fix (implemented in LinearInitEncoderWrapper)')
    print('=' * 70)
    print('  Pipeline data + y0/ystd → pure-scaled input → Wb reconstruction')
    print('  → pure-scaled output - x_mean/std_x → pipeline output')

    u_off = u_mean / std_u   # per-channel offset
    y_off = y0 / ystd
    x_off = x_mean / std_x

    x_hat_wrapper = np.zeros((M, NX_PHYS))
    for idx in range(M):
        k = hist_len + idx - 1
        # Pipeline data (what encoder receives)
        yh_pipe = ((y_t3[k - n: k + 1] - y0) / ystd).flatten()
        uh_pipe = ((u_t3[k - n: k + 1] - u_mean) / std_u).flatten()
        # Undo mean sub → pure-scaled (what Wb expects)
        yh_scaled = yh_pipe + np.tile(y_off, n + 1)
        uh_scaled = uh_pipe + np.tile(u_off, n + 1)
        # Reconstruct in pure-scaled
        x_hat_pure = Wb_psi_y @ yh_scaled + Wb_psi_u @ uh_scaled
        # Convert to pipeline convention
        x_hat_wrapper[idx] = x_hat_pure - x_off

    nrms_wrapper = nrms(x_hat_wrapper, x_true_norm)

    print(f'\n  Batch NRMS comparison:')
    print_state_table('', ['pipeline (broken)', 'wrapper fix', 'pure-scaled ref'],
                      [nrms_pipe, nrms_wrapper, nrms_scaled])

    match_pure = np.allclose(nrms_wrapper, nrms_scaled, atol=1e-6)
    print(f'\n  Wrapper matches pure-scaled quality: {"YES" if match_pure else "NO"}')
    if match_pure:
        print(f'  SUCCESS: The LinearInitEncoderWrapper convention fix works.')
        print(f'  The encoder will now achieve ~5-10% NRMS at initialization,')
        print(f'  limited by LTI model accuracy and O_n conditioning, not by')
        print(f'  normalization convention mismatch.')

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)

    print(f'\n  Pure-scaled reconstruction max NRMS:  {np.max(nrms_scaled):.4e}')
    print(f'  Pipeline (mean-sub) max NRMS:         {np.max(nrms_pipe):.4e}')
    print(f'  Wrapper fix max NRMS:                 {np.max(nrms_wrapper):.4e}')
    print(f'  Corrected (bias only) max NRMS:       {np.max(nrms_corrected):.4e}')
    print(f'  Reversed O_n max NRMS:                {np.max(nrms_rev):.4e}')

    wrapper_matches = np.max(np.abs(nrms_wrapper - nrms_scaled)) < 1e-4
    print(f'\n  Wrapper fix recovers pure-scaled quality?  {"YES" if wrapper_matches else "NO"}')
    print(f'  Time ordering is an issue?                 {"YES" if np.max(nrms_rev) < 0.1 * np.max(nrms_scaled) else "NO"}')

    if wrapper_matches:
        print(f'\n  FIX IMPLEMENTED: LinearInitEncoderWrapper now converts')
        print(f'    pipeline (mean-sub) → pure-scaled → Wb → pure-scaled → pipeline')
        print(f'  Remaining init NRMS ({np.max(nrms_scaled):.1%}) is the LTI model ceiling.')

    print('\n' + '=' * 70)
    print('Diagnostic complete.')


if __name__ == '__main__':
    main()
