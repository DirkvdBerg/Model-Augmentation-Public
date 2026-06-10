"""
diagnose_default_encoder.py — Find why the default encoder is 100x off.

Builds a model with the default (learned) encoder, trains it briefly,
then compares encoder output to analytical ground truth at each step.

Diagnostics:
  1. State range: what normalized state values does the data require?
  2. Random init: what does the encoder output before any training?
  3. After training: per-state error, correlation, scaling analysis
  4. Sensitivity: which states matter most for sim-RMS?
  5. Gradient magnitude: does the encoder receive useful gradients per state?

Run:
  conda run -n GraduationProject python scripts/gantry/verification/diagnose_default_encoder.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import deepSI
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model_augmentation.utils.utils import *
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.systems.gantry_ss import Cd, Dd, P

## ═══════════════════════════════════════════════════════════════════════════════
## Configuration — mirrors gantry_interconnect_dynamic.py
## ═══════════════════════════════════════════════════════════════════════════════

MODE = 'multisine'
NX_PHYS = 6
nu, ny = 3, 3
Y_OP = None
SEED = 42

FS_ORIG = 20000
FS_NEW  = 4000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

USE_F64  = False
DTYPE_NP = np.float64 if USE_F64 else np.float32
DTYPE_PT = torch.float64 if USE_F64 else torch.float32

NF_SECONDS   = 0.100
NANB_SECONDS = 0.030

HP = dict(
    NX_ANN=2,
    n_nodes_per_layer=64,
    n_hidden_layers=2,
    nf=max(1, int(NF_SECONDS / TS_NEW)),
    na_nb=max(1, int(NANB_SECONDS / TS_NEW)),
    batch_size=4000,
    lr=1e-4,
    epochs=50,      # short training for diagnostics
)

DIAG_EPOCHS = [0, 1, 5, 10, 25, 50]   # checkpoints to analyse

## ═══════════════════════════════════════════════════════════════════════════════
## Data loading
## ═══════════════════════════════════════════════════════════════════════════════

np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_SUBDIR = 'multisine' if MODE == 'multisine' else 'trajectories'
TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..',  '..',
                        'data', 'gantry', 'matlab', DATA_SUBDIR)

TRAIN_FILES = [
    'T1_Y_sweep_conservative.mat', 'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat',           'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat',        'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat',    'T8_X_sym_anti_Y_sweep.mat',
]
VAL_FILE  = 'V1_X_sym_Y_mid_sweep.mat'

def _load_u(d):
    if 'u_total' in d:
        return d['u_total']
    return d['u']

def load_traj(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    return deepSI.System_data(
        u=_load_u(d)[::D].astype(DTYPE_NP),
        y=d['y'][::D].astype(DTYPE_NP),
        dt=TS_NEW,
    )

train_list = [load_traj(f) for f in TRAIN_FILES]
train_data = deepSI.System_data_list(train_list)
val_data   = load_traj(VAL_FILE)

## ═══════════════════════════════════════════════════════════════════════════════
## Normalisation
## ═══════════════════════════════════════════════════════════════════════════════

u_all = np.concatenate([t.u for t in train_list])
y_all = np.concatenate([t.y for t in train_list])
fs = 1.0 / train_list[0].dt
P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)

x_logical_list = []
for t in train_list:
    pos_logical = (P_inv_T @ t.y.T).T
    vel_logical = np.diff(pos_logical, axis=0) * fs
    vel_logical = np.vstack([vel_logical[:1], vel_logical])
    x_logical_list.append(np.hstack([pos_logical, vel_logical]))
x_all = np.concatenate(x_logical_list)

x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
std_u  = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
ystd   = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
y0     = (Cd.numpy() @ x_mean.flatten()).astype(DTYPE_NP)

Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
Dd_np   = Dd.numpy()
PHY_IX  = np.arange(NX_PHYS)

## ═══════════════════════════════════════════════════════════════════════════════
## Diagnostic 1: State range analysis
## ═══════════════════════════════════════════════════════════════════════════════

x_norm_all = (x_all - x_mean.flatten()) / std_x.flatten()

state_names = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY']

print('\n' + '='*80)
print('  DIAGNOSTIC 1: Normalised state range across training data')
print('='*80)
print(f'  {"State":<10} {"min":>10} {"max":>10} {"mean":>10} {"std":>10}')
print('-'*55)
for i in range(NX_PHYS):
    s = x_norm_all[:, i]
    print(f'  {state_names[i]:<10} {s.min():>10.3f} {s.max():>10.3f} {s.mean():>10.3f} {s.std():>10.3f}')

print(f'\n  Physical state statistics (unnormalised):')
print(f'  {"State":<10} {"x_mean":>12} {"std_x":>12} {"unit":>8}')
print('-'*50)
units = ['m', 'rad', 'm', 'm/s', 'rad/s', 'm/s']
for i in range(NX_PHYS):
    print(f'  {state_names[i]:<10} {x_mean[i,0]:>12.6f} {std_x[i,0]:>12.6f} {units[i]:>8}')

print(f'\n  Output normalization:')
print(f'  {"Channel":<10} {"y0":>12} {"ystd":>12}')
print('-'*40)
ch_names = ['X1', 'X2', 'Y']
for i in range(ny):
    print(f'  {ch_names[i]:<10} {y0[i]:>12.6f} {ystd[i]:>12.6f}')

## ═══════════════════════════════════════════════════════════════════════════════
## Build model with DEFAULT encoder (not hybrid)
## ═══════════════════════════════════════════════════════════════════════════════

def build_model_default_encoder():
    NX_ANN = HP['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    na = HP['na_nb']
    nb = HP['na_nb']

    ic = Interconnect(nxd, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
    ).to(DTYPE_PT)
    out_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    ic.add_block(phy_block)
    ic.add_block(out_block)

    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=nxd,
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=torch.nn.Tanh,
    )
    ic.add_block(ann_block)

    ic.connect_block_signals(ann_block, ["x", "u"], ["xp"])
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", out_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_block, ["u"], ["y"])

    # DEFAULT encoder — do NOT inject hybrid
    fit_sys = SSE_Interconnect(
        interconnect=ic, na=na, nb=nb,
        e_net_kwargs={
            "n_nodes_per_layer": HP['n_nodes_per_layer'],
            "n_hidden_layers": HP['n_hidden_layers'],
        },
    )

    fit_sys.norm.u0   = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0   = y0
    fit_sys.norm.ystd = ystd

    fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
    for net in (fit_sys.encoder, fit_sys.hfn):
        net.to(DTYPE_PT)

    return fit_sys


## ═══════════════════════════════════════════════════════════════════════════════
## Analytical ground truth for encoder windows
## ═══════════════════════════════════════════════════════════════════════════════

def compute_analytical_x0(ypast_norm, upast_norm):
    """
    Given normalised past I/O windows, compute analytical physical states.
    Returns x0_phys_norm: (batch, NX_PHYS) in normalised coordinates.
    """
    batch = ypast_norm.shape[0]

    # Denormalise y
    y_denorm = ypast_norm.numpy() * ystd[None, None, :] + y0[None, None, :]  # (B, na, 3)

    # Stage -> logical coordinates
    pos_last = (P_inv_T @ y_denorm[:, -1, :].T).T   # (B, 3) [X, Theta, Y]
    pos_prev = (P_inv_T @ y_denorm[:, -2, :].T).T   # (B, 3)

    vel = (pos_last - pos_prev) * fs  # (B, 3) [dX, dTheta, dY]

    x_phys = np.hstack([pos_last, vel])  # (B, 6)
    x_phys_norm = (x_phys - x_mean.flatten()) / std_x.flatten()  # (B, 6)

    return x_phys_norm


## ═══════════════════════════════════════════════════════════════════════════════
## Extract encoder training windows
## ═══════════════════════════════════════════════════════════════════════════════

def get_encoder_windows(fit_sys, sys_data, n_windows=2000):
    """Extract (uhist, yhist, ufuture, yfuture) windows from data."""
    nf = HP['nf']
    na = HP['na_nb']
    nb = HP['na_nb']

    norm_data = fit_sys.norm.transform(sys_data)

    if hasattr(norm_data, 'sdl'):
        # System_data_list
        trajs = norm_data.sdl
    else:
        trajs = [norm_data]

    all_uhist, all_yhist = [], []
    for traj in trajs:
        u_arr = np.ascontiguousarray(traj.u)
        y_arr = np.ascontiguousarray(traj.y)
        T = len(u_arr)
        start = max(na, nb)
        for t in range(start, T - nf):
            all_uhist.append(u_arr[t-nb:t])
            all_yhist.append(y_arr[t-na:t])

    all_uhist = np.array(all_uhist, dtype=DTYPE_NP)
    all_yhist = np.array(all_yhist, dtype=DTYPE_NP)

    # Subsample if too many
    if len(all_uhist) > n_windows:
        idx = np.random.choice(len(all_uhist), n_windows, replace=False)
        all_uhist = all_uhist[idx]
        all_yhist = all_yhist[idx]

    return torch.tensor(all_uhist, dtype=DTYPE_PT), torch.tensor(all_yhist, dtype=DTYPE_PT)


## ═══════════════════════════════════════════════════════════════════════════════
## Diagnostic 2 & 3: Encoder output analysis
## ═══════════════════════════════════════════════════════════════════════════════

def analyse_encoder(fit_sys, uhist, yhist, x_gt, label):
    """Compare encoder output to analytical ground truth."""
    NX_ANN = HP['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    all_state_names = state_names + [f'x_ann{i}' for i in range(NX_ANN)]

    with torch.no_grad():
        x_enc = fit_sys.encoder(uhist, yhist).numpy()  # (B, nxd)

    x_enc_phys = x_enc[:, :NX_PHYS]  # (B, 6) physical states
    x_enc_ann  = x_enc[:, NX_PHYS:]  # (B, NX_ANN) augmented states

    print(f'\n{"="*80}')
    print(f'  ENCODER ANALYSIS: {label}')
    print(f'{"="*80}')

    # --- Per-state comparison ---
    print(f'\n  Physical states: encoder vs analytical ground truth')
    print(f'  {"State":<10} {"enc_mean":>10} {"gt_mean":>10} {"enc_std":>10} '
          f'{"gt_std":>10} {"MAE":>10} {"corr":>10} {"scale":>10}')
    print('-'*90)

    for i in range(NX_PHYS):
        enc_i = x_enc_phys[:, i]
        gt_i  = x_gt[:, i]
        mae = np.abs(enc_i - gt_i).mean()
        # Pearson correlation
        if enc_i.std() > 1e-10 and gt_i.std() > 1e-10:
            corr = np.corrcoef(enc_i, gt_i)[0, 1]
        else:
            corr = 0.0
        # Scale ratio: std(encoder) / std(ground_truth)
        scale = enc_i.std() / (gt_i.std() + 1e-10)

        print(f'  {state_names[i]:<10} {enc_i.mean():>10.4f} {gt_i.mean():>10.4f} '
              f'{enc_i.std():>10.4f} {gt_i.std():>10.4f} {mae:>10.4f} '
              f'{corr:>10.4f} {scale:>10.4f}')

    # --- Augmented state statistics ---
    print(f'\n  Augmented states (no ground truth):')
    print(f'  {"State":<10} {"mean":>10} {"std":>10} {"min":>10} {"max":>10}')
    print('-'*55)
    for i in range(NX_ANN):
        a = x_enc_ann[:, i]
        print(f'  {f"x_ann{i}":<10} {a.mean():>10.4f} {a.std():>10.4f} '
              f'{a.min():>10.4f} {a.max():>10.4f}')

    # --- Overall physical state error ---
    mse_phys = ((x_enc_phys - x_gt) ** 2).mean()
    rms_gt   = np.sqrt((x_gt ** 2).mean())
    ratio    = np.sqrt(mse_phys) / (rms_gt + 1e-10)
    print(f'\n  Overall physical state RMSE: {np.sqrt(mse_phys):.4f}')
    print(f'  Ground truth RMS:           {rms_gt:.4f}')
    print(f'  Error ratio:                {ratio:.2f}x')

    return x_enc_phys, x_enc_ann


## ═══════════════════════════════════════════════════════════════════════════════
## Diagnostic 4: Gradient magnitude per state
## ═══════════════════════════════════════════════════════════════════════════════

def analyse_gradients(fit_sys, uhist, yhist):
    """Check gradient magnitude from loss w.r.t. each encoder output state."""
    NX_ANN = HP['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    all_state_names = state_names + [f'x_ann{i}' for i in range(NX_ANN)]

    # Take a small batch
    B = min(64, uhist.shape[0])
    uh = uhist[:B].requires_grad_(False)
    yh = yhist[:B].requires_grad_(False)

    x_enc = fit_sys.encoder(uh, yh)  # (B, nxd)

    # Hook to capture gradient on encoder output
    grad_holder = [None]
    def hook(grad):
        grad_holder[0] = grad.detach().clone()
    x_enc.register_hook(hook)

    # Forward one step through interconnect (just to get a loss signal)
    # Use normalized data for ufuture/yfuture
    norm_data = fit_sys.norm.transform(val_data)
    na = HP['na_nb']
    u_future = torch.tensor(norm_data.u[na:na+1].astype(DTYPE_NP), dtype=DTYPE_PT).unsqueeze(0).expand(B, -1, -1)
    y_future = torch.tensor(norm_data.y[na:na+1].astype(DTYPE_NP), dtype=DTYPE_PT).unsqueeze(0).expand(B, -1, -1)

    # One-step rollout
    yhat, xp = fit_sys.hfn(x_enc, u_future[:, 0, :])
    loss = nn.functional.mse_loss(y_future[:, 0, :], yhat)
    loss.backward()

    print(f'\n{"="*80}')
    print(f'  DIAGNOSTIC: Gradient magnitude per encoder output state')
    print(f'{"="*80}')

    if grad_holder[0] is not None:
        grad = grad_holder[0].numpy()  # (B, nxd)
        print(f'  {"State":<10} {"|grad| mean":>12} {"|grad| std":>12} {"|grad| max":>12}')
        print('-'*55)
        for i in range(nxd):
            g = np.abs(grad[:, i])
            print(f'  {all_state_names[i]:<10} {g.mean():>12.2e} {g.std():>12.2e} {g.max():>12.2e}')
    else:
        print('  WARNING: No gradients captured (graph may be detached)')

    fit_sys.optimizer.zero_grad()


## ═══════════════════════════════════════════════════════════════════════════════
## Diagnostic 5: Sensitivity — what happens to sim output when we perturb x0?
## ═══════════════════════════════════════════════════════════════════════════════

def sensitivity_analysis(fit_sys):
    """
    Starting from analytical x0, perturb each state by +1 std and
    measure the output change. Shows which states the output is sensitive to.
    """
    NX_ANN = HP['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    na = HP['na_nb']
    all_state_names = state_names + [f'x_ann{i}' for i in range(NX_ANN)]

    norm_data = fit_sys.norm.transform(val_data)
    u_norm = torch.tensor(np.ascontiguousarray(norm_data.u), dtype=DTYPE_PT)

    # Analytical x0 for the first valid window
    y_stage = val_data.y
    pos_log = (P_inv_T @ y_stage.T).T
    vel_log = np.diff(pos_log, axis=0) * fs
    vel_log = np.vstack([vel_log[:1], vel_log])
    x_phys = np.hstack([pos_log, vel_log])

    x0_phys_norm = (x_phys[na] - x_mean.flatten()) / std_x.flatten()
    x0 = np.zeros(nxd, dtype=DTYPE_NP)
    x0[:NX_PHYS] = x0_phys_norm.astype(DTYPE_NP)

    # Simulate N_SIM steps from x0
    N_SIM = min(200, len(u_norm) - na - 1)

    def simulate_from_x0(x0_tensor):
        x = x0_tensor.unsqueeze(0)
        ys = []
        with torch.no_grad():
            for k in range(N_SIM):
                yhat, x = fit_sys.hfn(x, u_norm[na + k].unsqueeze(0))
                ys.append(yhat.numpy().flatten())
        return np.array(ys)  # (N_SIM, 3)

    # Baseline simulation
    x0_base = torch.tensor(x0, dtype=DTYPE_PT)
    y_base = simulate_from_x0(x0_base)

    print(f'\n{"="*80}')
    print(f'  DIAGNOSTIC: Output sensitivity to ±1σ perturbation of each state')
    print(f'  (RMS output change over {N_SIM} steps from analytical x0)')
    print(f'{"="*80}')
    print(f'  {"State":<10} {"Δy_RMS":>12} {"relative":>12}')
    print('-'*40)

    sensitivities = []
    for i in range(nxd):
        x0_pert = x0.copy()
        x0_pert[i] += 1.0  # +1 normalised std
        y_pert = simulate_from_x0(torch.tensor(x0_pert, dtype=DTYPE_PT))
        dy_rms = np.sqrt(((y_pert - y_base) ** 2).mean())
        sensitivities.append(dy_rms)

    # Normalise by max
    max_sens = max(sensitivities) + 1e-10
    for i in range(nxd):
        print(f'  {all_state_names[i]:<10} {sensitivities[i]:>12.2e} {sensitivities[i]/max_sens:>12.4f}')


## ═══════════════════════════════════════════════════════════════════════════════
## Diagnostic 6: Input informativity — is x0 distinguishable from past I/O?
## ═══════════════════════════════════════════════════════════════════════════════

def informativity_analysis(uhist, yhist, x_gt):
    """
    For each physical state, check whether different state values produce
    distinguishably different encoder input windows.

    Method: Fisher's discriminant ratio (FDR) per state.
      - Split windows into 'high' (above median) vs 'low' (below median)
        based on the ground truth state value.
      - Compute FDR = |mu_high - mu_low|^2 / (var_high + var_low)
        in the flattened input space.
      - High FDR → the input carries information about this state.
      - Low FDR → different state values produce similar inputs;
        no encoder can recover this state.

    Also reports a correlation-based measure: for each state, the max
    absolute correlation between that state and any single input feature.
    If this is near zero, the state is invisible in the raw input.
    """
    # Flatten inputs to (B, D) where D = 3*na + 3*nb
    inp = np.hstack([
        uhist.numpy().reshape(uhist.shape[0], -1),
        yhist.numpy().reshape(yhist.shape[0], -1),
    ])  # (B, D)

    B, D = inp.shape

    print(f'\n{"="*80}')
    print(f'  DIAGNOSTIC 6: Input informativity (can x0 be inferred from past I/O?)')
    print(f'  {B} windows, {D} input features')
    print(f'{"="*80}')

    print(f'\n  Fisher discriminant ratio (FDR): higher = more distinguishable')
    print(f'  Max |correlation| with any input feature: higher = more visible')
    print(f'\n  {"State":<10} {"FDR":>12} {"max|corr|":>12} {"verdict":>20}')
    print('-'*60)

    for i in range(NX_PHYS):
        gt_i = x_gt[:, i]
        median_i = np.median(gt_i)

        idx_lo = gt_i <= median_i
        idx_hi = gt_i > median_i

        if idx_lo.sum() < 10 or idx_hi.sum() < 10:
            print(f'  {state_names[i]:<10} {"N/A":>12} {"N/A":>12} {"too few samples":>20}')
            continue

        # FDR in input space
        mu_lo = inp[idx_lo].mean(axis=0)
        mu_hi = inp[idx_hi].mean(axis=0)
        var_lo = inp[idx_lo].var(axis=0)
        var_hi = inp[idx_hi].var(axis=0)

        # Sum across input dimensions (multivariate FDR)
        numerator = ((mu_hi - mu_lo) ** 2).sum()
        denominator = (var_lo + var_hi).sum() + 1e-10
        fdr = numerator / denominator

        # Max absolute correlation with any single input feature
        max_corr = 0.0
        gt_centered = gt_i - gt_i.mean()
        gt_norm = np.sqrt((gt_centered ** 2).sum()) + 1e-10
        for j in range(D):
            inp_centered = inp[:, j] - inp[:, j].mean()
            inp_norm = np.sqrt((inp_centered ** 2).sum()) + 1e-10
            corr_j = abs((gt_centered * inp_centered).sum() / (gt_norm * inp_norm))
            if corr_j > max_corr:
                max_corr = corr_j

        # Verdict
        if fdr > 1.0 and max_corr > 0.3:
            verdict = 'INFORMATIVE'
        elif fdr > 0.1 or max_corr > 0.1:
            verdict = 'marginal'
        else:
            verdict = 'NOT INFORMATIVE'

        print(f'  {state_names[i]:<10} {fdr:>12.4f} {max_corr:>12.4f} {verdict:>20}')

    print(f'\n  Interpretation:')
    print(f'    FDR > 1.0 + |corr| > 0.3  →  state is clearly encoded in past I/O')
    print(f'    FDR < 0.1 + |corr| < 0.1  →  state is invisible; encoder cannot learn it')
    print(f'    In between                 →  marginal; encoder may struggle')


## ═══════════════════════════════════════════════════════════════════════════════
## Main
## ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('\n' + '='*80)
    print('  DEFAULT ENCODER DIAGNOSTICS')
    print('='*80)
    print(f'  Mode:          {MODE}')
    print(f'  FS_NEW:        {FS_NEW} Hz')
    print(f'  NF_SECONDS:    {NF_SECONDS} s  →  nf = {HP["nf"]}')
    print(f'  NANB_SECONDS:  {NANB_SECONDS} s  →  na_nb = {HP["na_nb"]}')
    print(f'  NX_ANN:        {HP["NX_ANN"]}')
    print(f'  nxd:           {NX_PHYS + HP["NX_ANN"]}')
    print(f'  Encoder input: {3*HP["na_nb"] + 3*HP["na_nb"]} (= 3*(na+nb))')
    print(f'  Encoder arch:  simple_res_net, {HP["n_hidden_layers"]} hidden, {HP["n_nodes_per_layer"]} nodes, Tanh')
    print(f'  Epochs:        {HP["epochs"]}')

    # Build model
    fit_sys = build_model_default_encoder()

    # Extract validation windows
    print('\nExtracting encoder windows from validation data...')
    uhist_val, yhist_val = get_encoder_windows(fit_sys, val_data, n_windows=2000)
    print(f'  Got {uhist_val.shape[0]} windows, shape u={uhist_val.shape}, y={yhist_val.shape}')

    # Analytical ground truth
    x_gt = compute_analytical_x0(yhist_val, uhist_val)
    print(f'  Ground truth shape: {x_gt.shape}')

    # --- Diagnostic 2: Random init analysis ---
    x_enc_init, x_ann_init = analyse_encoder(
        fit_sys, uhist_val, yhist_val, x_gt, 'BEFORE TRAINING (random init)')

    # --- Diagnostic 4: Gradient analysis ---
    print('\nRunning gradient analysis...')
    analyse_gradients(fit_sys, uhist_val, yhist_val)

    # --- Diagnostic 5: Sensitivity analysis ---
    print('\nRunning sensitivity analysis...')
    sensitivity_analysis(fit_sys)

    # --- Diagnostic 6: Input informativity ---
    print('\nRunning informativity analysis...')
    informativity_analysis(uhist_val, yhist_val, x_gt)

    # --- Train and analyse at checkpoints ---
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                            'simulations', 'gantry_subnet', 'encoder_diagnostics')
    os.makedirs(save_dir, exist_ok=True)

    # Training with periodic encoder analysis
    epochs_done = 0
    checkpoint_results = {}

    for target_epoch in DIAG_EPOCHS:
        if target_epoch == 0:
            continue  # already analysed
        epochs_to_run = target_epoch - epochs_done
        if epochs_to_run <= 0:
            continue

        print(f'\n--- Training epochs {epochs_done+1} to {target_epoch} ---')
        fit_sys.fit(
            train_sys_data=train_data, val_sys_data=val_data,
            batch_size=HP['batch_size'], epochs=epochs_to_run,
            auto_fit_norm=False,
            loss_kwargs={'nf': HP['nf']},
            optimizer_kwargs={'lr': HP['lr']},
            validation_measure="sim-RMS",
        )
        epochs_done = target_epoch

        x_enc, x_ann = analyse_encoder(
            fit_sys, uhist_val, yhist_val, x_gt, f'AFTER {target_epoch} EPOCHS')

        # Store for plotting
        checkpoint_results[target_epoch] = {
            'x_enc': x_enc.copy(),
            'x_ann': x_ann.copy(),
        }

    # --- Summary plot ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Default encoder: per-state error convergence', fontsize=13)

    epochs_list = sorted(checkpoint_results.keys())
    for i in range(NX_PHYS):
        ax = axes[i // 3, i % 3]
        maes = []
        for ep in epochs_list:
            enc_i = checkpoint_results[ep]['x_enc'][:, i]
            mae = np.abs(enc_i - x_gt[:, i]).mean()
            maes.append(mae)
        ax.plot(epochs_list, maes, 'b.-', linewidth=1.5, markersize=8)
        ax.set_title(f'{state_names[i]}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE (normalised)')
        ax.grid(True)
        ax.set_yscale('log')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'encoder_convergence.png')
    plt.savefig(plot_path, dpi=150)
    print(f'\nSaved convergence plot: {plot_path}')

    # --- Scatter plots: encoder vs ground truth (final epoch) ---
    if epochs_list:
        final = checkpoint_results[epochs_list[-1]]
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
        fig2.suptitle(f'Encoder vs analytical ground truth (epoch {epochs_list[-1]})', fontsize=13)
        for i in range(NX_PHYS):
            ax = axes2[i // 3, i % 3]
            ax.scatter(x_gt[:, i], final['x_enc'][:, i], s=1, alpha=0.3)
            lims = [
                min(x_gt[:, i].min(), final['x_enc'][:, i].min()),
                max(x_gt[:, i].max(), final['x_enc'][:, i].max()),
            ]
            ax.plot(lims, lims, 'r--', linewidth=1)
            ax.set_xlabel(f'{state_names[i]} (analytical)')
            ax.set_ylabel(f'{state_names[i]} (encoder)')
            ax.set_title(state_names[i])
            ax.grid(True)
            ax.set_aspect('equal', adjustable='datalim')

        plt.tight_layout()
        plot_path2 = os.path.join(save_dir, 'encoder_scatter.png')
        plt.savefig(plot_path2, dpi=150)
        print(f'Saved scatter plot: {plot_path2}')

    print('\n' + '='*80)
    print('  DIAGNOSTICS COMPLETE')
    print('='*80)
