"""
diag7_native_fix_parity.py
---------------------------
Verify that the D-017 convention fix baked natively into linear_encoder_init_aug
is bit-identical to the _FixedEncoder wrapper used in diag6.

Encoders under test (all built from the same normalized LTI, same seed):
  A         : linear_encoder_init_aug without fix  (no stats passed)
  B_wrapper : _FixedEncoder(linear_encoder_init_aug, stats)  -- diag6 approach
  B_native  : linear_encoder_init_aug(... u_mean=..., y0=..., ...)  -- new native fix

Checks:
  P1: B_wrapper forward == B_native forward on random batch  (atol 1e-6)
  P2: B_wrapper forward == B_native forward on real window data  (atol 1e-6)
  P3: B_native mean pos NRMS < A mean pos NRMS / 5  (fix still effective natively)
  P4: B_native mean pos NRMS < 0.50  (absolute quality gate)
  P5: |B_native vel NRMS - B_wrapper vel NRMS| < 1e-5  (training target unchanged)

Same LTI setup as diag6 (seed=42, U_OFFSET, normalize_linear_ss_matrices) so
results are directly comparable.

Usage:
    conda run -n GraduationProject python \\
        scripts/gantry/encoder-augmentation/diag7_native_fix_parity.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.utils.utils import normalize_linear_ss_matrices
from deepSI.system_data import System_data

# =============================================================================
# Config -- identical to diag6
# =============================================================================
NX_PHYS  = 6
NX_AUG   = 0
NU       = 3
NY       = 3
NA = NB  = 25
SEED     = 42
DTYPE_NP = np.float32
DTYPE_PT = torch.float32

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']
POS_IDX     = [0, 1, 2]
VEL_IDX     = [3, 4, 5]

T_SYN    = 4000
N_TRAIN  = 3
U_OFFSET = np.array([2.0, -1.5, 1.5])
U_SIGMA  = np.array([0.8,  0.6, 0.7])

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet',
                       'diagnostics', 'diag7')
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# _FixedEncoder -- identical to diag6 (wrapper approach, reference)
# =============================================================================

class _FixedEncoder(nn.Module):
    """diag6 wrapper approach -- kept as reference for parity check."""

    def __init__(self, enc, u_mean, std_u, y0, ystd, x_mean, std_x):
        super().__init__()
        self.enc = enc
        u_off = np.tile((u_mean / std_u).flatten(), NB + 1).astype(DTYPE_NP)
        y_off = np.tile((y0    / ystd  ).flatten(), NA + 1).astype(DTYPE_NP)
        x_off = (x_mean / std_x).flatten().astype(DTYPE_NP)
        self.register_buffer('u_off', torch.tensor(u_off).view(-1, 1))
        self.register_buffer('y_off', torch.tensor(y_off).view(-1, 1))
        self.register_buffer('x_off', torch.tensor(x_off).view(-1, 1))

    def forward(self, uhist, yhist):
        B = uhist.size(0)
        uh3 = uhist.view(B, NU * (NB + 1), 1)
        yh3 = yhist.view(B, NY * (NA + 1), 1)
        uh_fix = uh3 + self.u_off
        yh_fix = yh3 + self.y_off
        enc = self.enc
        x_b = enc.Wb_psi_u @ uh_fix + enc.Wb_psi_y @ yh_fix
        x_a = enc.Wa_psi_u @ uh_fix + enc.Wa_psi_y @ yh_fix
        x_b = x_b - self.x_off
        x = torch.cat([x_b, x_a], dim=1).view(B, NX_PHYS + NX_AUG)
        if not enc.flag_linear_only:
            x = x + enc.net(torch.cat([uhist.view(B, -1), yhist.view(B, -1)], dim=1))
        return x


# =============================================================================
# Synthetic data -- identical to diag6
# =============================================================================

def make_random_stable_lti(rng):
    A_raw = rng.standard_normal((NX_PHYS, NX_PHYS))
    rho   = np.max(np.abs(np.linalg.eigvals(A_raw)))
    A = A_raw / (rho + 0.1) * 0.90
    B = rng.standard_normal((NX_PHYS, NU)) * 0.5
    C = rng.standard_normal((NY, NX_PHYS)) * 0.5
    D = np.zeros((NY, NU))
    return A, B, C, D


def simulate_traj(rng, A, B, C, D, T, x0):
    u = rng.standard_normal((T, NU)) * U_SIGMA + U_OFFSET
    x = np.empty((T, NX_PHYS))
    x[0] = x0
    for t in range(1, T):
        x[t] = A @ x[t - 1] + B @ u[t - 1]
    y = (C @ x.T).T + (D @ u.T).T
    return u.astype(DTYPE_NP), y.astype(DTYPE_NP), x.astype(DTYPE_NP)


def make_windows(u_norm, y_norm, x_norm, na, nb):
    T = len(u_norm)
    uh, yh, xg = [], [], []
    for t in range(nb, T):
        uh.append(u_norm[t - nb : t + 1].flatten())
        yh.append(y_norm[t - na : t + 1].flatten())
        xg.append(x_norm[t])
    return (np.array(uh, DTYPE_NP), np.array(yh, DTYPE_NP),
            np.array(xg, DTYPE_NP))


# =============================================================================
# Evaluation helpers
# =============================================================================

def eval_encoder_nrms(encoder, uh_np, yh_np, xgt_np, batch=512):
    x_hats = []
    with torch.no_grad():
        for i in range(0, len(uh_np), batch):
            uh = torch.tensor(uh_np[i:i+batch], dtype=DTYPE_PT)
            yh = torch.tensor(yh_np[i:i+batch], dtype=DTYPE_PT)
            x_hats.append(encoder(uh, yh).numpy())
    x_hat = np.concatenate(x_hats, axis=0)
    diff  = x_hat - xgt_np
    nrms  = (np.sqrt(np.mean(diff**2, axis=0))
             / (np.sqrt(np.mean(xgt_np**2, axis=0)) + 1e-8))
    return nrms


# =============================================================================
# Check helper
# =============================================================================
results = {}

def check(name, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    results[name] = {'status': status, 'detail': detail}
    marker = '  ' if condition else '!!'
    print(f'  [{status}] {marker} {name}' + (f'  ({detail})' if detail else ''))
    return condition


# =============================================================================
# Main
# =============================================================================

def main():
    print('=' * 70)
    print('diag7: Native D-017 fix parity check')
    print('  B_wrapper: _FixedEncoder(linear_encoder_init_aug, stats)  [diag6]')
    print('  B_native:  linear_encoder_init_aug(..., u_mean=..., ...)   [new]')
    print(f'  NX_AUG={NX_AUG}, NA=NB={NA}, SEED={SEED}')
    print('  Same LTI setup as diag6 -- results directly comparable')
    print('=' * 70)

    rng = np.random.default_rng(SEED)

    # -------------------------------------------------------------------------
    # 1. LTI + data (identical to diag6)
    # -------------------------------------------------------------------------
    print('\n--- Building random stable LTI (diag6 setup) ---', flush=True)
    A_d, B_d, C_d, D_d = make_random_stable_lti(rng)
    rho = np.max(np.abs(np.linalg.eigvals(A_d)))
    print(f'  spectral radius = {rho:.6f}')

    x_eq = np.linalg.solve(np.eye(NX_PHYS) - A_d, B_d @ U_OFFSET)
    train_trajs = [
        simulate_traj(rng, A_d, B_d, C_d, D_d, T_SYN,
                      x_eq + 0.01 * rng.standard_normal(NX_PHYS))
        for _ in range(N_TRAIN)
    ]
    val_traj = simulate_traj(rng, A_d, B_d, C_d, D_d, T_SYN,
                              x_eq + 0.01 * rng.standard_normal(NX_PHYS))

    u_all = np.concatenate([u for u, _, _ in train_trajs])
    y_all = np.concatenate([y for _, y, _ in train_trajs])
    x_all = np.concatenate([x for _, _, x in train_trajs])

    sys_data = System_data(u=u_all.astype(np.float64),
                           y=y_all.astype(np.float64),
                           x=x_all.astype(np.float64))
    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        A_d, B_d, C_d, D_d, sys_data, state_ix=np.arange(NX_PHYS)
    )
    Ad_bar = Ad_bar.astype(DTYPE_NP)
    Bd_bar = Bd_bar.astype(DTYPE_NP)
    Cd_bar = Cd_bar.astype(DTYPE_NP)
    Dd_bar = Dd_bar.astype(DTYPE_NP)

    u_mean = u_all.mean(axis=0).astype(DTYPE_NP)
    std_u  = u_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    y0     = y_all.mean(axis=0).astype(DTYPE_NP)
    ystd   = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    x_mean = x_all.mean(axis=0).astype(DTYPE_NP)
    std_x  = x_all.std(axis=0).astype(DTYPE_NP) + 1e-8

    print(f'  ||y0/ystd||_inf   = {np.max(np.abs(y0/ystd)):.4f}')
    print(f'  ||u_m/std_u||_inf = {np.max(np.abs(u_mean/std_u)):.4f}')

    # -------------------------------------------------------------------------
    # 2. Build encoders A, B_wrapper, B_native (identical W^b, same seed)
    # -------------------------------------------------------------------------
    print('\n--- Building encoders ---', flush=True)
    enc_kwargs = dict(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=NU, ny=NY, na=NA, nb=NB,
        nx_aug=NX_AUG,
        n_nodes_per_layer=64, n_hidden_layers=2,
        flag_linear_only=False,
    )

    torch.manual_seed(SEED)
    enc_A = linear_encoder_init_aug(**enc_kwargs)

    torch.manual_seed(SEED)
    enc_B_base = linear_encoder_init_aug(**enc_kwargs)
    enc_B_wrapper = _FixedEncoder(enc_B_base, u_mean, std_u, y0, ystd, x_mean, std_x)

    torch.manual_seed(SEED)
    enc_B_native = linear_encoder_init_aug(
        **enc_kwargs,
        u_mean=u_mean, std_u=std_u,
        y0=y0, ystd=ystd,
        x_mean=x_mean, std_x=std_x,
    )

    print(f'  fix_enabled on B_native: {enc_B_native.fix_enabled}')
    print(f'  fix_enabled on A:        {enc_A.fix_enabled}')

    # -------------------------------------------------------------------------
    # 3. Parity check on random batch (machine precision)
    # -------------------------------------------------------------------------
    print('\n======================================================================')
    print('PART 1: Parity on random batch  (no data needed)')
    print('======================================================================')

    torch.manual_seed(99)
    uh_rand = torch.randn(256, NU * (NB + 1), dtype=DTYPE_PT)
    yh_rand = torch.randn(256, NY * (NA + 1), dtype=DTYPE_PT)

    with torch.no_grad():
        out_wrapper = enc_B_wrapper(uh_rand, yh_rand)
        out_native  = enc_B_native(uh_rand, yh_rand)

    max_diff_rand = (out_wrapper - out_native).abs().max().item()
    check('P1: B_wrapper == B_native on random batch',
          max_diff_rand < 1e-6,
          f'max_diff={max_diff_rand:.2e}')

    # -------------------------------------------------------------------------
    # 4. Parity check on real window data
    # -------------------------------------------------------------------------
    print('\n======================================================================')
    print('PART 2: Parity on real window data from val trajectory')
    print('======================================================================')

    u_v, y_v, x_v = val_traj
    un_v = (u_v - u_mean) / std_u
    yn_v = (y_v - y0) / ystd
    xn_v = (x_v - x_mean) / std_x

    uh_vl, yh_vl, xg_vl = make_windows(un_v, yn_v, xn_v, NA, NB)
    print(f'  Val windows: {len(uh_vl)}')

    with torch.no_grad():
        uh_t = torch.tensor(uh_vl, dtype=DTYPE_PT)
        yh_t = torch.tensor(yh_vl, dtype=DTYPE_PT)
        out_w2 = enc_B_wrapper(uh_t, yh_t).numpy()
        out_n2 = enc_B_native(uh_t, yh_t).numpy()

    max_diff_data = np.abs(out_w2 - out_n2).max()
    check('P2: B_wrapper == B_native on window data',
          max_diff_data < 1e-6,
          f'max_diff={max_diff_data:.2e}')

    # -------------------------------------------------------------------------
    # 5. NRMS: native fix vs no fix (should replicate diag6 Part 1)
    # -------------------------------------------------------------------------
    print('\n======================================================================')
    print('PART 3: NRMS check -- B_native vs A  (replicates diag6 Part 1)')
    print('======================================================================')

    nrms_A       = eval_encoder_nrms(enc_A,        uh_vl, yh_vl, xg_vl)
    nrms_wrapper = eval_encoder_nrms(enc_B_wrapper, uh_vl, yh_vl, xg_vl)
    nrms_native  = eval_encoder_nrms(enc_B_native,  uh_vl, yh_vl, xg_vl)

    print(f'\n  {"Channel":<6}  {"NRMS A":>10}  {"NRMS B_wrap":>12}  '
          f'{"NRMS B_native":>14}  {"wrap==native":>12}')
    print(f'  {"-"*6}  {"-"*10}  {"-"*12}  {"-"*14}  {"-"*12}')
    for i, name in enumerate(STATE_NAMES):
        match = abs(nrms_wrapper[i] - nrms_native[i]) < 1e-5
        print(f'  {name:<6}  {nrms_A[i]:>10.6f}  {nrms_wrapper[i]:>12.6f}  '
              f'{nrms_native[i]:>14.6f}  {"OK" if match else "MISMATCH":>12}')

    mean_pos_A      = nrms_A[POS_IDX].mean()
    mean_pos_native = nrms_native[POS_IDX].mean()
    mean_vel_wrap   = nrms_wrapper[VEL_IDX].mean()
    mean_vel_native = nrms_native[VEL_IDX].mean()

    print(f'\n  Mean pos NRMS:  A={mean_pos_A:.6f}   B_native={mean_pos_native:.6f}')
    print(f'  Mean vel NRMS:  B_wrapper={mean_vel_wrap:.6f}   B_native={mean_vel_native:.6f}')

    print('\n--- Part 3 checks ---')
    check('P3: B_native pos NRMS < A / 5',
          mean_pos_native < mean_pos_A / 5,
          f'A={mean_pos_A:.6f} B_native={mean_pos_native:.6f}')
    check('P4: B_native mean pos NRMS < 0.50',
          mean_pos_native < 0.50,
          f'{mean_pos_native:.6f}')
    check('P5: |B_native vel - B_wrapper vel| < 1e-5',
          abs(mean_vel_native - mean_vel_wrap) < 1e-5,
          f'B_wrapper={mean_vel_wrap:.2e}  B_native={mean_vel_native:.2e}')

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    n_pass = sum(1 for v in results.values() if v['status'] == 'PASS')
    n_fail = sum(1 for v in results.values() if v['status'] == 'FAIL')
    print(f'\n{"="*50}')
    print(f'  {n_pass}/{len(results)} checks passed  ({n_fail} failed)')

    out_json = os.path.join(OUT_DIR, 'diag7_results.json')
    with open(out_json, 'w') as f:
        json.dump({
            'config': {
                'NX_AUG': NX_AUG, 'NA': NA, 'NB': NB, 'SEED': SEED,
                'T_SYN': T_SYN, 'N_TRAIN': N_TRAIN,
            },
            'parity': {
                'max_diff_random_batch': float(max_diff_rand),
                'max_diff_window_data':  float(max_diff_data),
            },
            'nrms': {
                'A':        nrms_A.tolist(),
                'wrapper':  nrms_wrapper.tolist(),
                'native':   nrms_native.tolist(),
            },
            'checks': {k: v for k, v in results.items()},
        }, f, indent=2)
    print(f'  Results saved to {out_json}')


if __name__ == '__main__':
    main()
