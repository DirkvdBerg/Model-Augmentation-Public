"""
diag6_encoder_fix_comparison.py
---------------------------------
Compare linear_encoder_init_aug WITHOUT and WITH the normalization convention
fix (D-017) at both initialisation and after short training.

Background:
  normalize_linear_ss_matrices normalizes the LTI matrices using np.std()
  (pure scaling by standard deviation) from the training data.  The W^b
  matrix therefore expects PURE-SCALED I/O: y/ystd, u/std_u -> x/std_x.

  The deepSI pipeline, however, feeds MEAN-SUBTRACTED inputs:
    (y - y0)/ystd,  (u - u_mean)/std_u
  and expects:
    (x - x_mean)/std_x

  This introduces a constant bias when y0 != 0 or u_mean != 0 (diag5:
  ||bias_x|| = 0.3129, position channels 7000-21000x worse on real data).

  The D-017 fix:
    Before W^b: add y0/ystd  and  u_mean/std_u  -> pure-scaled I/O
    After  W^b: subtract x_mean/std_x            -> pipeline convention

Correct encoder initialization workflow:
  1. normalize_linear_ss_matrices(A,B,C,D, train_data) -> A_bar,B_bar,C_bar,D_bar
  2. linear_encoder_init_aug(A_bar, B_bar, C_bar, D_bar, ...)
     => W^b maps y/ystd -> x/std_x  (pure-scaled)
     => Pipeline feeds (y-y0)/ystd  (mean-subtracted)  -- MISMATCH

Encoders under test:
  A: linear_encoder_init_aug(A_bar,...) -- no fix  (biased initialization)
  B: same encoder wrapped in _FixedEncoder  (D-017 convention fix)
  Both NX_AUG=0, identical ANN seed, identical W^b matrices.
  Only forward() differs.

System for test:
  Random STABLE LTI (NX=6, NU=3, NY=3), spectral radius ~ 0.9.
  Non-zero operating point via constant U_OFFSET forcing (x_eq, y_eq != 0).
  This gives non-trivial y0/ystd, u_mean/std_u, x_mean/std_x -- exactly
  the regime where the fix matters.  Avoids gantry marginal stability issues.
  Algebraic property tested is independent of specific LTI matrices.

Checks:
  S1: Encoder A mean pos NRMS > 5x B   (fix is meaningful)
  S2: Encoder B mean pos NRMS < 0.50   (fix is effective at init)
  S3: |A vel NRMS - B vel NRMS| < 0.30 (fix doesn't hurt velocities)
  T1: B epoch-0 val loss < A            (better init -> lower initial loss)
  T2: B final pos NRMS <= A * 2.0      (B stays competitive after training)
  T3: Report whether training improved/worsened A position NRMS vs init

Usage:
    conda run -n GraduationProject python \\
        scripts/gantry/encoder-augmentation/diag6_encoder_fix_comparison.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.utils.utils import normalize_linear_ss_matrices
from deepSI.system_data import System_data

# =============================================================================
# Config
# =============================================================================
NX_PHYS  = 6
NX_AUG   = 0       # clean comparison: fix effect on W^b only
NU       = 3
NY       = 3
NA = NB  = 25      # encoder history
NF       = 15      # rollout steps for training loss
N_EPOCHS = 10
LR       = 1e-4
BATCH    = 64
SEED     = 42
DTYPE_NP = np.float32
DTYPE_PT = torch.float32

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']
POS_IDX     = [0, 1, 2]
VEL_IDX     = [3, 4, 5]

# Synthetic data config -- random stable LTI with non-zero equilibrium
# This cleanly tests the algebraic property without gantry marginal-stability issues.
T_SYN    = 4000     # samples per trajectory
N_TRAIN  = 3        # number of train trajectories
# HEURISTIC: large enough U_OFFSET relative to U_SIGMA so u_mean/std_u >> 1
# and the equilibrium x_eq = (I-A)^{-1} B U_OFFSET is visible in x_mean/std_x.
U_OFFSET = np.array([2.0, -1.5, 1.5])   # constant forcing -> non-zero equilibrium
U_SIGMA  = np.array([0.8,  0.6, 0.7])   # excitation std  -> u_mean/std_u >> 1

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet',
                       'diagnostics', 'diag6')
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# _FixedEncoder  (local wrapper -- diagnostic only, never touches model_augmentation)
# =============================================================================

class _FixedEncoder(nn.Module):
    """
    Wraps linear_encoder_init_aug to apply the D-017 normalization convention fix.

    The encoder's W^b (built from normalize_linear_ss_matrices output) maps
    y/ystd, u/std_u -> x/std_x  (pure-scaled convention).

    The pipeline feeds (y-y0)/ystd, (u-u_mean)/std_u and expects (x-x_mean)/std_x.

    Fix:
      Before W^b: add y0/ystd  and  u_mean/std_u  (-> pure-scaled inputs)
      After  W^b: subtract x_mean/std_x            (-> pipeline convention output)
    ANN correction receives original (mean-subtracted) inputs -- D-017 intent.

    NOTE: diagnostic wrapper only. Implement in pre_encoder.py after conclusion.
    """

    def __init__(self, enc, u_mean, std_u, y0, ystd, x_mean, std_x):
        super().__init__()
        self.enc = enc

        # THEORY: linearity of W^b (Hoekstra 2026 Eq. 16-17), D-017.
        u_off = np.tile((u_mean / std_u).flatten(), NB + 1).astype(DTYPE_NP)
        y_off = np.tile((y0    / ystd  ).flatten(), NA + 1).astype(DTYPE_NP)
        x_off = (x_mean / std_x).flatten().astype(DTYPE_NP)

        self.register_buffer('u_off', torch.tensor(u_off).view(-1, 1))  # (nu*(nb+1), 1)
        self.register_buffer('y_off', torch.tensor(y_off).view(-1, 1))  # (ny*(na+1), 1)
        self.register_buffer('x_off', torch.tensor(x_off).view(-1, 1))  # (nx, 1)

    def forward(self, uhist, yhist):
        B = uhist.size(0)

        uh3 = uhist.view(B, NU * (NB + 1), 1)
        yh3 = yhist.view(B, NY * (NA + 1), 1)

        # Add offsets: (mean-subtracted) -> pure-scaled
        uh_fix = uh3 + self.u_off
        yh_fix = yh3 + self.y_off

        enc = self.enc
        x_b = enc.Wb_psi_u @ uh_fix + enc.Wb_psi_y @ yh_fix   # (B, nx, 1)
        x_a = enc.Wa_psi_u @ uh_fix + enc.Wa_psi_y @ yh_fix   # (B, 0, 1)

        # Subtract offset: x/std_x -> (x-x_mean)/std_x  (pipeline convention)
        x_b = x_b - self.x_off   # (B, nx, 1)

        x = torch.cat([x_b, x_a], dim=1).view(B, NX_PHYS + NX_AUG)  # (B, 6)

        if not enc.flag_linear_only:
            x = x + enc.net(
                torch.cat([uhist.view(B, -1), yhist.view(B, -1)], dim=1)
            )
        return x


# =============================================================================
# Synthetic data generation
# =============================================================================

def make_random_stable_lti(rng):
    """
    Random stable discrete LTI with NX=6, NU=3, NY=3.
    Spectral radius ~ 0.9 (stable, good reconstructability).
    """
    A_raw = rng.standard_normal((NX_PHYS, NX_PHYS))
    rho   = np.max(np.abs(np.linalg.eigvals(A_raw)))
    A = A_raw / (rho + 0.1) * 0.90   # spectral radius ~ 0.9
    B = rng.standard_normal((NX_PHYS, NU)) * 0.5
    C = rng.standard_normal((NY, NX_PHYS)) * 0.5
    D = np.zeros((NY, NU))            # no feedthrough (typical for position output)
    return A, B, C, D


def simulate_traj(rng, A, B, C, D, T, x0):
    """
    Simulate stable LTI from x0 with excitation around U_OFFSET.
    Returns (u, y, x) float32 arrays, shape (T, nu/ny/nx).
    """
    u = rng.standard_normal((T, NU)) * U_SIGMA + U_OFFSET
    x = np.empty((T, NX_PHYS))
    x[0] = x0
    for t in range(1, T):
        x[t] = A @ x[t - 1] + B @ u[t - 1]
    y = (C @ x.T).T + (D @ u.T).T
    return u.astype(DTYPE_NP), y.astype(DTYPE_NP), x.astype(DTYPE_NP)


# =============================================================================
# Window construction
# =============================================================================

def make_windows(u_norm, y_norm, x_norm, na, nb, nf):
    """
    Build sliding-window arrays from one normalised trajectory.
    Returns:
      uh  (N, (nb+1)*nu)  encoder input windows
      yh  (N, (na+1)*ny)  encoder output windows
      uf  (N, nf, nu)     future inputs  [u[t], ..., u[t+nf-1]]
      yf  (N, nf, ny)     future outputs [y[t+1], ..., y[t+nf]]  (rollout targets)
      xg  (N, nx)         ground-truth state x[t] (normalised)
    """
    T = len(u_norm)
    uh, yh, uf, yf, xg = [], [], [], [], []
    for t in range(nb, T - nf):
        uh.append(u_norm[t - nb : t + 1].flatten())
        yh.append(y_norm[t - na : t + 1].flatten())
        uf.append(u_norm[t     : t + nf])
        yf.append(y_norm[t + 1 : t + nf + 1])
        xg.append(x_norm[t])
    return (np.array(uh, DTYPE_NP), np.array(yh, DTYPE_NP),
            np.array(uf, DTYPE_NP), np.array(yf, DTYPE_NP),
            np.array(xg, DTYPE_NP))


# =============================================================================
# Evaluation helpers
# =============================================================================

def eval_encoder_nrms(encoder, uh_np, yh_np, xgt_np, batch=512):
    """Per-channel NRMS of encoder output vs ground-truth normalised state."""
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
    return nrms   # (6,)


def run_linear_rollout(encoder, A_t, B_t, C_t, uhist, yhist, u_fut):
    """
    NF-step linear rollout in pipeline convention.

    State update and output in zero-mean (pipeline) space:
      x_norm[t+k+1] = A_bar @ x_norm[t+k] + B_bar @ u_norm[t+k]
      y_norm[t+k+1] = C_bar @ x_norm[t+k+1]   (D=0)
    Advance state FIRST, then compute output (matches target yf = y[t+1:t+nf+1]).
    """
    x = encoder(uhist, yhist)   # (B, 6)  pipeline convention
    y_preds = []
    for k in range(NF):
        u_k = u_fut[:, k, :]            # (B, nu)  u[t+k]
        x   = x @ A_t.T + u_k @ B_t.T  # (B, nx)  x[t+k+1]
        y_k = x @ C_t.T                 # (B, ny)  y[t+k+1]  (D=0)
        y_preds.append(y_k)
    return torch.stack(y_preds, dim=1)  # (B, NF, ny)


def eval_val_loss(encoder, A_t, B_t, C_t,
                  uh_np, yh_np, uf_np, yf_np, batch=256):
    total, n_b = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(uh_np), batch):
            uh = torch.tensor(uh_np[i:i+batch], dtype=DTYPE_PT)
            yh = torch.tensor(yh_np[i:i+batch], dtype=DTYPE_PT)
            uf = torch.tensor(uf_np[i:i+batch], dtype=DTYPE_PT)
            yf = torch.tensor(yf_np[i:i+batch], dtype=DTYPE_PT)
            yp = run_linear_rollout(encoder, A_t, B_t, C_t, uh, yh, uf)
            total += F.mse_loss(yp, yf).item()
            n_b   += 1
    return total / max(n_b, 1)


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
    print('diag6: Normalization fix encoder comparison (D-017)')
    print('  Encoder A: no fix   Encoder B: D-017 fix (_FixedEncoder wrapper)')
    print(f'  NX_AUG={NX_AUG}, NA=NB={NA}, NF={NF}, N_EPOCHS={N_EPOCHS}')
    print('  System: random stable LTI (NX=6, NU=3, NY=3, rho~0.9)')
    print('=' * 70)

    rng = np.random.default_rng(SEED)

    # -------------------------------------------------------------------------
    # 1. Random stable LTI system
    # -------------------------------------------------------------------------
    print('\n--- Building random stable LTI ---', flush=True)
    A_d, B_d, C_d, D_d = make_random_stable_lti(rng)
    rho = np.max(np.abs(np.linalg.eigvals(A_d)))
    print(f'  spectral radius = {rho:.6f}')

    # Forced equilibrium (non-zero x_eq from constant U_OFFSET)
    # HEURISTIC: U_OFFSET chosen so x_eq is well away from 0
    x_eq = np.linalg.solve(np.eye(NX_PHYS) - A_d, B_d @ U_OFFSET)
    y_eq = C_d @ x_eq   # equilibrium output
    print(f'  U_OFFSET = {U_OFFSET}')
    print(f'  x_eq     = {x_eq.round(4)}')
    print(f'  y_eq     = {y_eq.round(4)}')

    # -------------------------------------------------------------------------
    # 2. Generate synthetic trajectories (start near equilibrium)
    # -------------------------------------------------------------------------
    print('\n--- Generating synthetic data ---', flush=True)
    train_trajs = [
        simulate_traj(rng, A_d, B_d, C_d, D_d, T_SYN,
                      x_eq + 0.01 * rng.standard_normal(NX_PHYS))
        for _ in range(N_TRAIN)
    ]
    val_traj = simulate_traj(rng, A_d, B_d, C_d, D_d, T_SYN,
                              x_eq + 0.01 * rng.standard_normal(NX_PHYS))
    print(f'  {N_TRAIN} train + 1 val trajectories, {T_SYN} samples each')

    u_all  = np.concatenate([u for u, _, _ in train_trajs])
    y_all  = np.concatenate([y for _, y, _ in train_trajs])
    x_all  = np.concatenate([x for _, _, x in train_trajs])

    # -------------------------------------------------------------------------
    # 3. Normalize LTI matrices with training data statistics
    #    normalize_linear_ss_matrices uses np.std (pure std, same as ystd below).
    #    W^b then maps y/ystd -> x/std_x  (no mean subtraction = pure scaling).
    # -------------------------------------------------------------------------
    print('\n--- Normalizing LTI matrices ---', flush=True)
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
    rho_bar = np.max(np.abs(np.linalg.eigvals(Ad_bar)))
    print(f'  spectral radius (Ad_bar) = {rho_bar:.6f}  '
          f'(same as A_d -- similarity transform)')

    # -------------------------------------------------------------------------
    # 4. Pipeline normalization constants
    #    Convention: pipeline feeds (y-y0)/ystd, (u-u_mean)/std_u
    #    expects  (x-x_mean)/std_x
    #    These MATCH what normalize_linear_ss_matrices used.
    # -------------------------------------------------------------------------
    u_mean = u_all.mean(axis=0).astype(DTYPE_NP)
    std_u  = u_all.std(axis=0).astype(DTYPE_NP)  + 1e-8
    y0     = y_all.mean(axis=0).astype(DTYPE_NP)
    ystd   = y_all.std(axis=0).astype(DTYPE_NP)  + 1e-8
    x_mean = x_all.mean(axis=0).astype(DTYPE_NP)
    std_x  = x_all.std(axis=0).astype(DTYPE_NP)  + 1e-8

    print(f'  x_mean        = {x_mean.round(4)}')
    print(f'  std_x         = {std_x.round(4)}')
    print(f'  y0            = {y0.round(4)}')
    print(f'  ystd          = {ystd.round(4)}')
    print(f'  u_mean        = {u_mean.round(4)}')
    print(f'  std_u         = {std_u.round(4)}')
    print(f'  ||y0/ystd||_inf   = {np.max(np.abs(y0/ystd)):.4f}')
    print(f'  ||u_m/std_u||_inf = {np.max(np.abs(u_mean/std_u)):.4f}')

    # -------------------------------------------------------------------------
    # 5. Build encoders A and B (both from NORMALIZED matrices)
    # -------------------------------------------------------------------------
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
    enc_B = _FixedEncoder(enc_B_base, u_mean, std_u, y0, ystd, x_mean, std_x)

    n_params = sum(p.numel() for p in enc_A.parameters())
    print(f'\n  Encoder parameters: {n_params}  (ANN seed={SEED})')

    # -------------------------------------------------------------------------
    # 6. Linear rollout matrices (pipeline convention)
    #    In zero-mean pipeline space: x[t+1] = Ad_bar @ x[t] + Bd_bar @ u[t]
    #    Same eigenvalues as A_d (similarity transform -> same stability).
    # -------------------------------------------------------------------------
    A_t = torch.tensor(Ad_bar, dtype=DTYPE_PT)
    B_t = torch.tensor(Bd_bar, dtype=DTYPE_PT)
    C_t = torch.tensor(Cd_bar, dtype=DTYPE_PT)

    # -------------------------------------------------------------------------
    # 7. Build window datasets
    # -------------------------------------------------------------------------
    print('\n--- Building window datasets ---', flush=True)

    def normalise(u, y, x):
        return ((u - u_mean) / std_u,
                (y - y0)    / ystd,
                (x - x_mean) / std_x)

    train_wins = [[], [], [], [], []]
    for u, y, x in train_trajs:
        un, yn, xn = normalise(u, y, x)
        for k, arr in enumerate(make_windows(un, yn, xn, NA, NB, NF)):
            train_wins[k].append(arr)
    uh_tr, yh_tr, uf_tr, yf_tr, xg_tr = [np.concatenate(w) for w in train_wins]

    u_v, y_v, x_v = val_traj
    un_v, yn_v, xn_v = normalise(u_v, y_v, x_v)
    uh_vl, yh_vl, uf_vl, yf_vl, xg_vl = make_windows(un_v, yn_v, xn_v, NA, NB, NF)

    print(f'  Train: {len(uh_tr)} windows    Val: {len(uh_vl)} windows')

    # =========================================================================
    # PART 1: Static NRMS at initialisation
    # =========================================================================
    print('\n' + '=' * 70)
    print('PART 1: Static NRMS at initialisation  (no training)')
    print('=' * 70)
    print('  Ground truth: x_norm = (x - x_mean) / std_x  (pipeline convention)')
    print('  Encoder A sees (y-y0)/ystd but W^b expects y/ystd  -> BIAS')
    print('  Encoder B adds y0/ystd, u_mean/std_u before W^b   -> CORRECTED')

    nrms_A_init = eval_encoder_nrms(enc_A, uh_vl, yh_vl, xg_vl)
    nrms_B_init = eval_encoder_nrms(enc_B, uh_vl, yh_vl, xg_vl)

    print(f'\n  {"Channel":<6}  {"NRMS A (no fix)":>16}  {"NRMS B (fix)":>14}  '
          f'{"ratio A/B":>10}')
    print(f'  {"-"*6}  {"-"*16}  {"-"*14}  {"-"*10}')
    for i, name in enumerate(STATE_NAMES):
        ratio = nrms_A_init[i] / (nrms_B_init[i] + 1e-12)
        print(f'  {name:<6}  {nrms_A_init[i]:>16.4f}  {nrms_B_init[i]:>14.4f}  '
              f'{ratio:>10.2f}')

    mean_pos_A = nrms_A_init[POS_IDX].mean()
    mean_pos_B = nrms_B_init[POS_IDX].mean()
    mean_vel_A = nrms_A_init[VEL_IDX].mean()
    mean_vel_B = nrms_B_init[VEL_IDX].mean()

    print(f'\n  Mean position NRMS:  A={mean_pos_A:.4f}   B={mean_pos_B:.4f}')
    print(f'  Mean velocity NRMS:  A={mean_vel_A:.4f}   B={mean_vel_B:.4f}')

    print('\n--- Part 1 checks ---')
    check('S1: pos NRMS A > 5x B',
          mean_pos_A > 5 * mean_pos_B,
          f'A={mean_pos_A:.4f} B={mean_pos_B:.4f}')
    check('S2: Encoder B mean pos NRMS < 0.50', mean_pos_B < 0.50,
          f'{mean_pos_B:.4f}')
    check('S3: |vel NRMS A - B| < 0.30',
          abs(mean_vel_A - mean_vel_B) < 0.30,
          f'A={mean_vel_A:.4f} B={mean_vel_B:.4f}')

    # =========================================================================
    # PART 2: Short training comparison  (linear rollout, stable dynamics)
    # =========================================================================
    print('\n' + '=' * 70)
    print(f'PART 2: Training comparison  ({N_EPOCHS} epochs, LR={LR})')
    print('  Linear rollout on stable Ad_bar, Bd_bar, Cd_bar (no divergence)')
    print('=' * 70)

    opt_A = torch.optim.Adam(enc_A.parameters(), lr=LR)
    opt_B = torch.optim.Adam(enc_B.parameters(), lr=LR)

    n_train = len(uh_tr)
    idx     = np.arange(n_train)

    hist = {'epoch': [], 'loss_A': [], 'loss_B': [], 'nrms_A': [], 'nrms_B': []}

    loss_A_0 = eval_val_loss(enc_A, A_t, B_t, C_t, uh_vl, yh_vl, uf_vl, yf_vl)
    loss_B_0 = eval_val_loss(enc_B, A_t, B_t, C_t, uh_vl, yh_vl, uf_vl, yf_vl)
    nr_A_0   = eval_encoder_nrms(enc_A, uh_vl, yh_vl, xg_vl)
    nr_B_0   = eval_encoder_nrms(enc_B, uh_vl, yh_vl, xg_vl)

    hist['epoch'].append(0)
    hist['loss_A'].append(loss_A_0); hist['loss_B'].append(loss_B_0)
    hist['nrms_A'].append(nr_A_0.tolist()); hist['nrms_B'].append(nr_B_0.tolist())

    print(f'\n  Epoch 0  loss_A={loss_A_0:.4e}  loss_B={loss_B_0:.4e}')
    print(f'           pos_NRMS_A={np.mean(nr_A_0[POS_IDX]):.4f}  '
          f'pos_NRMS_B={np.mean(nr_B_0[POS_IDX]):.4f}')

    for epoch in range(1, N_EPOCHS + 1):
        np.random.shuffle(idx)
        enc_A.train(); enc_B.train()

        for start in range(0, n_train - BATCH + 1, BATCH):
            bi = idx[start : start + BATCH]
            uh = torch.tensor(uh_tr[bi], dtype=DTYPE_PT)
            yh = torch.tensor(yh_tr[bi], dtype=DTYPE_PT)
            uf = torch.tensor(uf_tr[bi], dtype=DTYPE_PT)
            yf = torch.tensor(yf_tr[bi], dtype=DTYPE_PT)

            opt_A.zero_grad()
            F.mse_loss(run_linear_rollout(enc_A, A_t, B_t, C_t, uh, yh, uf),
                       yf).backward()
            opt_A.step()

            opt_B.zero_grad()
            F.mse_loss(run_linear_rollout(enc_B, A_t, B_t, C_t, uh, yh, uf),
                       yf).backward()
            opt_B.step()

        enc_A.eval(); enc_B.eval()
        vl_A = eval_val_loss(enc_A, A_t, B_t, C_t, uh_vl, yh_vl, uf_vl, yf_vl)
        vl_B = eval_val_loss(enc_B, A_t, B_t, C_t, uh_vl, yh_vl, uf_vl, yf_vl)
        nr_A = eval_encoder_nrms(enc_A, uh_vl, yh_vl, xg_vl)
        nr_B = eval_encoder_nrms(enc_B, uh_vl, yh_vl, xg_vl)

        hist['epoch'].append(epoch)
        hist['loss_A'].append(vl_A); hist['loss_B'].append(vl_B)
        hist['nrms_A'].append(nr_A.tolist()); hist['nrms_B'].append(nr_B.tolist())

        if epoch % 5 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}  loss_A={vl_A:.4e}  loss_B={vl_B:.4e}'
                  f'  pos_NRMS_A={np.mean(nr_A[POS_IDX]):.4f}'
                  f'  pos_NRMS_B={np.mean(nr_B[POS_IDX]):.4f}',
                  flush=True)

    nrms_A_final = np.array(hist['nrms_A'][-1])
    nrms_B_final = np.array(hist['nrms_B'][-1])
    mean_pos_A_f = nrms_A_final[POS_IDX].mean()
    mean_pos_B_f = nrms_B_final[POS_IDX].mean()

    print('\n--- Part 2 checks ---')
    check('T1: B epoch-0 val loss < A', loss_B_0 < loss_A_0,
          f'A={loss_A_0:.4e} B={loss_B_0:.4e}')
    check('T2: B final pos NRMS <= A * 2.0',
          mean_pos_B_f <= mean_pos_A_f * 2.0,
          f'A={mean_pos_A_f:.4f} B={mean_pos_B_f:.4f}')

    nrms_A_init_pos = np.array(hist['nrms_A'][0])[POS_IDX].mean()
    nrms_A_fin_pos  = nrms_A_final[POS_IDX].mean()
    t3_label = 'improved' if nrms_A_fin_pos < nrms_A_init_pos else 'worsened'
    check('T3: A pos NRMS after training (report only)', True,
          f'init={nrms_A_init_pos:.4f} -> final={nrms_A_fin_pos:.4f} ({t3_label})')

    # =========================================================================
    # Plots
    # =========================================================================
    epochs_arr = np.array(hist['epoch'])
    nrms_A_arr = np.array(hist['nrms_A'])
    nrms_B_arr = np.array(hist['nrms_B'])

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    x_pos = np.arange(NX_PHYS)
    w = 0.38
    ax1.bar(x_pos - w/2, nrms_A_init, w, label='A: no fix',   color='C3', alpha=0.85)
    ax1.bar(x_pos + w/2, nrms_B_init, w, label='B: with fix', color='C0', alpha=0.85)
    ax1.set_xticks(x_pos); ax1.set_xticklabels(STATE_NAMES)
    ax1.set_ylabel('NRMS vs (x-x_mean)/std_x')
    ax1.set_title('Part 1: Encoder NRMS at init -- D-017 fix improves state recovery?')
    ax1.legend(); ax1.grid(True, axis='y', alpha=0.3)
    fig1.tight_layout()
    p1 = os.path.join(OUT_DIR, 'diag6_init_nrms.png')
    fig1.savefig(p1, dpi=120); plt.close(fig1)
    print(f'\n  Saved {p1}')

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes2[0]
    ax.semilogy(epochs_arr, hist['loss_A'], 'C3-o', ms=3, label='A: no fix')
    ax.semilogy(epochs_arr, hist['loss_B'], 'C0-s', ms=3, label='B: with fix')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val MSE loss (linear rollout)')
    ax.set_title('Validation loss over training')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes2[1]
    pos_nrms_A = nrms_A_arr[:, POS_IDX].mean(axis=1)
    pos_nrms_B = nrms_B_arr[:, POS_IDX].mean(axis=1)
    ax.semilogy(epochs_arr, pos_nrms_A, 'C3-o', ms=3, label='A: no fix')
    ax.semilogy(epochs_arr, pos_nrms_B, 'C0-s', ms=3, label='B: with fix')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Mean position NRMS')
    ax.set_title('Position state recovery over training')
    ax.legend(); ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    p2 = os.path.join(OUT_DIR, 'diag6_training_curves.png')
    fig2.savefig(p2, dpi=120); plt.close(fig2)
    print(f'  Saved {p2}')

    # =========================================================================
    # Summary
    # =========================================================================
    n_pass = sum(1 for v in results.values() if v['status'] == 'PASS')
    n_fail = sum(1 for v in results.values() if v['status'] == 'FAIL')
    print(f'\n{"="*50}')
    print(f'  {n_pass}/{len(results)} checks passed  ({n_fail} failed)')

    save = {
        'config': {
            'NX_AUG': NX_AUG, 'NA': NA, 'NB': NB, 'NF': NF,
            'N_EPOCHS': N_EPOCHS, 'LR': LR,
            'SEED': SEED, 'T_SYN': T_SYN, 'N_TRAIN': N_TRAIN,
            'system': 'random stable LTI (rho~0.9)',
            'note': 'normalize_linear_ss_matrices applied; linear rollout for Part 2',
        },
        'normalization': {
            'y0_over_ystd_inf': float(np.max(np.abs(y0/ystd))),
            'umean_over_stdu_inf': float(np.max(np.abs(u_mean/std_u))),
            'xmean_over_stdx_inf': float(np.max(np.abs(x_mean/std_x))),
        },
        'part1_static': {
            'nrms_A_init':      nrms_A_init.tolist(),
            'nrms_B_init':      nrms_B_init.tolist(),
            'mean_pos_nrms_A':  float(mean_pos_A),
            'mean_pos_nrms_B':  float(mean_pos_B),
            'mean_vel_nrms_A':  float(mean_vel_A),
            'mean_vel_nrms_B':  float(mean_vel_B),
        },
        'part2_training': {
            'epochs':       hist['epoch'],
            'loss_A':       hist['loss_A'],
            'loss_B':       hist['loss_B'],
            'nrms_A':       hist['nrms_A'],
            'nrms_B':       hist['nrms_B'],
            'loss_A_epoch0': float(loss_A_0),
            'loss_B_epoch0': float(loss_B_0),
            'mean_pos_nrms_A_final': float(mean_pos_A_f),
            'mean_pos_nrms_B_final': float(mean_pos_B_f),
        },
        'checks': {k: v for k, v in results.items()},
    }
    out_json = os.path.join(OUT_DIR, 'diag6_results.json')
    with open(out_json, 'w') as f:
        json.dump(save, f, indent=2)
    print(f'  Results saved to {out_json}')


if __name__ == '__main__':
    main()
