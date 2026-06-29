"""diag20_na_sweep.py

Sweeps na=nb over [17, 25, 50, 100, 200] and validates encoder quality at each value.

Three validation steps per na=nb value (no training):
  1. Observability  -- O_n noise amplification per channel (analytical)
  2. Encoder R2     -- empirical R2_raw + R2_linmap for all physical channels
  3. Init sim-NRMS  -- initial simulation error from encoder-init (apply_experiment)

Outputs (to diagnostics/):
  diag20_observability.png   -- noise amplification vs na, per channel
  diag20_encoder_r2.png      -- R2_raw + R2_linmap for velocity channels vs na
  diag20_init_nrms.png       -- init sim-NRMS per output channel vs na
  diag20_results.npz
"""

import os
import sys
import numpy as np
import torch
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
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
from model_augmentation.utils.utils import normalize_linear_ss_matrices

## ===================================================================
## Configuration
## ===================================================================

MODE     = 'multisine'
NX_PHYS  = 6
NX_ANN   = 2
nu, ny   = 3, 3
Y_OP     = None
SEED     = 42

FS_ORIG  = 20000
FS_NEW   = 4000
D        = FS_ORIG // FS_NEW
TS_NEW   = 1.0 / FS_NEW

USE_F64  = False
DTYPE_NP = np.float64 if USE_F64 else np.float32
DTYPE_PT = torch.float64 if USE_F64 else torch.float32

NA_VALUES   = [17, 25, 50, 100, 200]   # sweep values
NA_RIGHT    = 1                          # encoder right-side extension (linear_map convention)
UP_SAMPLE   = 2
N_NODES     = 16
N_HIDDEN    = 2

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'diagnostics')
os.makedirs(SAVE_DIR, exist_ok=True)
RID = 'diag20'

## ===================================================================
## Data loading
## ===================================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry', 'matlab', MODE)
TRAIN_FILES = [
    'T1_Y_sweep_conservative.mat', 'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat', 'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat', 'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat', 'T8_X_sym_anti_Y_sweep.mat',
]
VAL_FILE = 'V1_X_sym_Y_mid_sweep.mat'

def _load_u(d):
    return d['u_total'] if 'u_total' in d else d['u']

def load_traj(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    return deepSI.System_data(u=_load_u(d)[::D].astype(DTYPE_NP),
                              y=d['y'][::D].astype(DTYPE_NP), dt=TS_NEW)

def load_mat_aug(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    x_logical = d['x_logical'][::D].astype(DTYPE_NP)
    return x_logical

train_list    = [load_traj(f) for f in TRAIN_FILES]
train_data    = deepSI.System_data_list(train_list)
val_data      = load_traj(VAL_FILE)
val_x_logical = load_mat_aug(VAL_FILE)
print(f'Loaded data. Val: {val_data.y.shape}  x_logical: {val_x_logical.shape}')

## ===================================================================
## Normalisation
## ===================================================================

u_all = np.concatenate([t.u for t in train_list])
y_all = np.concatenate([t.y for t in train_list])
fs    = 1.0 / train_list[0].dt
P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)

x_logical_list = []
for t in train_list:
    pos = (P_inv_T @ t.y.T).T
    vel = np.diff(pos, axis=0) * fs
    vel = np.vstack([vel[:1], vel])
    x_logical_list.append(np.hstack([pos, vel]))
x_all = np.concatenate(x_logical_list)

x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
std_u  = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
ystd   = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
y0     = y_all.mean(axis=0).astype(DTYPE_NP)

Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
Dd_np   = Dd.numpy()
PHY_IX  = np.arange(NX_PHYS)
nxd     = NX_PHYS + NX_ANN

x_true_norm = (val_x_logical.astype(DTYPE_NP) - x_mean.flatten()) / std_x.flatten()

# Normalized FP matrices (built once, shared across all na values)
Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)
baseline_npz_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry',
                                  'baseline_simulations', f'{MODE}_LPV', 'baseline_states.npz')
if os.path.exists(baseline_npz_path):
    bl = np.load(baseline_npz_path, allow_pickle=True)
    x_phys_all = np.concatenate(bl['x_train_phys'])
else:
    x_phys_all = x_all
    print("WARNING: baseline_states.npz not found, using finite-diff states")

sys_data_with_x = deepSI.System_data(u=u_all, y=y_all)
sys_data_with_x.x = x_phys_all
Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(Ad, Bd, Cd_dt, Dd_dt, sys_data_with_x)

labels_phys   = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']
vel_channels  = [3, 4, 5]   # dq1, dq2, dq3 indices
labels_vel    = ['dq1', 'dq2', 'dq3']
t_val         = np.arange(len(val_data.y)) * val_data.dt

## ===================================================================
## Helper: build full model with a given na=nb
## ===================================================================

def build_model_na(na):
    """Build SSE_Interconnect with given na=nb (overrides _get_encoder_dims)."""
    nb = na
    ic = Interconnect(nxd, nu, ny, debugging=False)
    phy_block = Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u,
                                   x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
                                   up_sample=UP_SAMPLE).to(DTYPE_PT)
    C_aug_init = np.zeros((ny, NX_ANN), dtype=DTYPE_NP)
    C_aug_init[2, 0] = 1e-2
    out_phys = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    out_aug  = Parameterized_Linear_Output_Block(C=C_aug_init,
                D=np.zeros((ny, nu), dtype=DTYPE_NP), flag_loss_reg=False)
    AUG_IX = np.arange(NX_PHYS, nxd)
    ann_block = Static_ANN_Block(nz=nxd + nu, nw=NX_ANN,
                                 n_nodes_per_layer=N_NODES, n_hidden_layers=N_HIDDEN,
                                 net=zero_init_feed_forward_nn, activation=torch.nn.Tanh)
    for blk in [phy_block, out_phys, out_aug, ann_block]:
        ic.add_block(blk)
    ic.connect_block_signals(ann_block, ["x", "u"], [])
    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(AUG_IX, nxd))
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", out_phys, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_phys, ["u"], ["y"])
    ic.connect_signals("x", out_aug, "concat", selection_matrix(AUG_IX, nxd))
    ic.connect_block_signals(out_aug, ["u"], ["y"])

    fit_sys = SSE_Interconnect(interconnect=ic, na=na, nb=nb,
                               na_right=NA_RIGHT, nb_right=NA_RIGHT,
                               e_net_kwargs={"n_nodes_per_layer": N_NODES,
                                             "n_hidden_layers": N_HIDDEN})
    fit_sys.norm.u0 = u_mean.flatten(); fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0 = y0;              fit_sys.norm.ystd = ystd

    fit_sys.encoder = linear_encoder_init_aug(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
        nx_aug=NX_ANN,
        n_nodes_per_layer=N_NODES, n_hidden_layers=N_HIDDEN,
        flag_linear_only=False,
        u_mean=u_mean, std_u=std_u, y0=y0, ystd=ystd,
        x_mean=x_mean, std_x=std_x,
    ).to(DTYPE_PT)

    fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
    fit_sys.hfn.to(DTYPE_PT)
    fit_sys.eval()
    return fit_sys

## ===================================================================
## Sweep
## ===================================================================

res_noise_amp  = []   # (n_na, NX_PHYS)
res_r2_raw     = []   # (n_na, NX_PHYS)
res_r2_lin     = []   # (n_na, NX_PHYS)
res_nrms_init  = []   # (n_na, ny)
res_cond       = []   # (n_na,)

print(f'\nSweeping na=nb over {NA_VALUES}')
print('='*60)

for na in NA_VALUES:
    print(f'\n--- na=nb={na}  (window={na/FS_NEW*1000:.1f} ms) ---')

    # ── Part 1: Observability ──────────────────────────────────────
    O_n = np.zeros(((na + 1) * ny, NX_PHYS), dtype=np.float64)
    A64, C64 = Ad_bar.astype(np.float64), Cd_bar.astype(np.float64)
    for i in range(na + 1):
        O_n[i * ny:(i + 1) * ny, :] = C64 @ np.linalg.matrix_power(A64, i)
    _, S_svd, _ = np.linalg.svd(O_n, full_matrices=False)
    O_pinv = np.linalg.pinv(O_n)
    noise_amp = np.linalg.norm(O_pinv, axis=1)   # (NX_PHYS,)
    cond = S_svd[0] / S_svd[-1]
    res_noise_amp.append(noise_amp)
    res_cond.append(cond)
    print(f'  O_n cond={cond:.1f}  noise_amp: '
          + '  '.join(f'{labels_phys[ch]}={noise_amp[ch]:.2f}' for ch in range(NX_PHYS)))

    # ── Part 2: Encoder R2 ─────────────────────────────────────────
    fit_sys = build_model_na(na)
    val_norm = fit_sys.norm.transform(val_data)
    yn = np.ascontiguousarray(val_norm.y, dtype=DTYPE_NP)
    un = np.ascontiguousarray(val_norm.u, dtype=DTYPE_NP)
    N  = len(yn)
    k0 = na + 1
    stride = max(1, (N - k0) // 2000)
    k_ix = np.arange(k0, N, stride)
    ypast = np.stack([yn[k - na : k + NA_RIGHT] for k in k_ix])
    upast = np.stack([un[k - na : k + NA_RIGHT] for k in k_ix])
    with torch.no_grad():
        x_hat = fit_sys.encoder(
            torch.tensor(upast, dtype=DTYPE_PT),
            torch.tensor(ypast, dtype=DTYPE_PT),
        ).numpy()
    x_hat_phys = x_hat[:, :NX_PHYS]
    xt = x_true_norm[k_ix]

    def _r2(ref, est):
        ss_res = ((ref - est)**2).sum(axis=0)
        ss_tot = ((ref - ref.mean(axis=0))**2).sum(axis=0)
        return 1.0 - ss_res / (ss_tot + 1e-12)

    r2_raw = _r2(xt, x_hat_phys)

    A_aug = np.hstack([x_hat_phys, np.ones((len(x_hat_phys), 1), dtype=DTYPE_NP)])
    W, *_ = np.linalg.lstsq(A_aug, xt, rcond=None)
    r2_lin = _r2(xt, A_aug @ W)

    res_r2_raw.append(r2_raw)
    res_r2_lin.append(r2_lin)
    print(f'  R2_raw:  ' + '  '.join(f'{labels_phys[ch]}={r2_raw[ch]:+.4f}' for ch in range(NX_PHYS)))
    print(f'  R2_lin:  ' + '  '.join(f'{labels_phys[ch]}={r2_lin[ch]:+.4f}' for ch in range(NX_PHYS)))

    # ── Part 3: Init sim-NRMS ──────────────────────────────────────
    sim_result = fit_sys.apply_experiment(val_data)
    cheat_n    = sim_result.cheat_n
    y_hat      = sim_result.y
    y_ref      = val_data.y
    nrms = np.sqrt(((y_hat[cheat_n:] - y_ref[cheat_n:])**2).mean(axis=0)) / ystd
    res_nrms_init.append(nrms)
    print(f'  Init sim-NRMS: X1={nrms[0]:.4f}  X2={nrms[1]:.4f}  Y={nrms[2]:.4f}')

    del fit_sys  # free memory between iterations

res_noise_amp = np.array(res_noise_amp)   # (n_na, NX_PHYS)
res_r2_raw    = np.array(res_r2_raw)
res_r2_lin    = np.array(res_r2_lin)
res_nrms_init = np.array(res_nrms_init)
res_cond      = np.array(res_cond)
NA_ARR        = np.array(NA_VALUES)

## ===================================================================
## Summary table
## ===================================================================

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'  {"na":>4s}  {"cond":>8s}  {"amp_dq2":>10s}  '
      f'{"R2raw_dq2":>10s}  {"R2lin_dq2":>10s}  '
      f'{"NRMS_X1":>10s}  {"NRMS_X2":>10s}')
for i, na in enumerate(NA_VALUES):
    print(f'  {na:4d}  {res_cond[i]:8.1f}  {res_noise_amp[i,4]:10.2f}  '
          f'{res_r2_raw[i,4]:+10.4f}  {res_r2_lin[i,4]:+10.4f}  '
          f'{res_nrms_init[i,0]:10.4f}  {res_nrms_init[i,1]:10.4f}')

## ===================================================================
## Plots
## ===================================================================

# Plot 1: noise amplification vs na per channel
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 4))
for ch in range(NX_PHYS):
    ls = '-' if ch >= 3 else '--'
    ax1a.semilogy(NA_ARR, res_noise_amp[:, ch], marker='o', ms=5, ls=ls,
                  label=labels_phys[ch])
ax1a.set_xlabel('na=nb'); ax1a.set_ylabel('Noise amplification (log)')
ax1a.set_title('Per-channel noise amplification vs na\n(lower = easier to reconstruct)')
ax1a.legend(fontsize=8); ax1a.grid(True, which='both')

ax1b.semilogy(NA_ARR, res_cond, 'ko-', ms=5)
ax1b.set_xlabel('na=nb'); ax1b.set_ylabel('Condition number of O_n (log)')
ax1b.set_title('Observability matrix condition number vs na')
ax1b.grid(True, which='both')

fig1.suptitle('diag20 -- Observability analysis vs na=nb')
fig1.tight_layout()
fig1.savefig(os.path.join(SAVE_DIR, f'{RID}_observability.png'), dpi=150)
print(f'\nSaved: {RID}_observability.png')

# Plot 2: encoder R2 for velocity channels vs na
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
for ch_ix, ch in enumerate(vel_channels):
    axes2[0].plot(NA_ARR, res_r2_raw[:, ch], marker='o', ms=5, label=labels_phys[ch])
    axes2[1].plot(NA_ARR, res_r2_lin[:, ch], marker='o', ms=5, label=labels_phys[ch])
for ax in axes2:
    ax.axhline(1.0, color='k', lw=0.5, ls=':')
    ax.set_xlabel('na=nb'); ax.legend(fontsize=8); ax.grid(True)
    ax.set_ylim([-0.1, 1.05])
axes2[0].set_ylabel('R2_raw'); axes2[0].set_title('R2_raw (velocity channels) vs na')
axes2[1].set_ylabel('R2_linmap'); axes2[1].set_title('R2_linmap (velocity channels) vs na')
fig2.suptitle('diag20 -- Encoder velocity reconstruction quality vs na=nb')
fig2.tight_layout()
fig2.savefig(os.path.join(SAVE_DIR, f'{RID}_encoder_r2.png'), dpi=150)
print(f'Saved: {RID}_encoder_r2.png')

# Plot 3: init sim-NRMS vs na
ch_labels_out = ['X1', 'X2', 'Y']
fig3, ax3 = plt.subplots(figsize=(8, 4))
for ch, lbl in enumerate(ch_labels_out):
    ax3.semilogy(NA_ARR, res_nrms_init[:, ch], marker='o', ms=5, label=lbl)
ax3.set_xlabel('na=nb'); ax3.set_ylabel('Init sim-NRMS (log)')
ax3.set_title('diag20 -- Initial sim-NRMS vs na=nb (no training)\n'
              'Target: approach baseline FP level (X1~0.0034, X2~0.0031)')
ax3.axhline(0.0034, color='C0', ls=':', lw=1.2, label='Baseline FP X1')
ax3.axhline(0.0031, color='C1', ls=':', lw=1.2, label='Baseline FP X2')
ax3.axhline(0.0037, color='C2', ls=':', lw=1.2, label='Baseline FP Y')
ax3.legend(fontsize=8); ax3.grid(True, which='both')
fig3.tight_layout()
fig3.savefig(os.path.join(SAVE_DIR, f'{RID}_init_nrms.png'), dpi=150)
print(f'Saved: {RID}_init_nrms.png')

plt.close('all')

np.savez(os.path.join(SAVE_DIR, f'{RID}_results.npz'),
         NA_VALUES=NA_ARR,
         noise_amp=res_noise_amp, cond=res_cond,
         r2_raw=res_r2_raw, r2_lin=res_r2_lin,
         nrms_init=res_nrms_init,
         labels_phys=np.array(labels_phys))
print(f'Saved: {RID}_results.npz')

print('\n' + '='*60)
print('diag20 complete')
print('='*60)
