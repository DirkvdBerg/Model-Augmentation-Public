"""diag19_init_sim_analysis.py

Diagnoses WHY the init sim-NRMS is 12x worse than baseline FP for X1/X2.

Three questions:
  1. OBSERVABILITY — Is dq2 fundamentally harder to reconstruct?
     Compute O_n SVD and per-channel noise amplification of the pseudoinverse.

  2. ENCODER BIAS — Is the dq2 estimation error systematic (fixable) or random?
     Plot x_hat[ch] - x_true_norm[ch] over time for all 6 physical channels.

  3. ORACLE-INIT HFN — Is the issue the encoder, or the HFN model itself?
     Run HFN from true x_logical (x_aug=0), compare NRMS to encoder-init and baseline FP.

     If oracle-init HFN ≈ baseline FP  -> HFN model is fine, problem is encoder quality
     If oracle-init HFN ≈ encoder-init -> problem is HFN model, not encoder

Outputs (to diagnostics/):
  diag19_observability.png    -- O_n singular values + per-channel noise amplification
  diag19_encoder_bias.png     -- encoder residual per channel over time
  diag19_nrms_comparison.png  -- bar chart: encoder-init vs oracle-init vs baseline FP
  diag19_results.npz
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
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn, HybridGantryEncoder
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
from model_augmentation.utils.utils import normalize_linear_ss_matrices

## ==============================================================================═
## Configuration — copied verbatim from gantry_interconnect_dynamic.py
## ==============================================================================═

MODE = 'multisine'
NX_PHYS = 6
nu  = 3
ny  = 3
Y_OP = None
ENCODER_INIT = 'linear_map'
ANN_ACTIVATION = 'tanh'
SEED = 42

FS_ORIG = 20000
FS_NEW  = 4000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

USE_F64  = False
DTYPE_NP = np.float64 if USE_F64 else np.float32
DTYPE_PT = torch.float64 if USE_F64 else torch.float32

DEFAULT_HP = dict(
    NX_ANN=2,
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=2,
    nf=max(1, int(0.100 / TS_NEW)),
    na_nb=0,
    batch_size=256,
    lr=1e-4,
    epochs=0,
)
DEFAULT_HP['na_nb'] = (NX_PHYS + DEFAULT_HP['NX_ANN']) * 2 + 1

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'diagnostics')
os.makedirs(SAVE_DIR, exist_ok=True)
RID = 'diag19'

## ==============================================================================═
## Data loading — copied verbatim
## ==============================================================================═

np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_SUBDIR = 'multisine'
TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry', 'matlab', DATA_SUBDIR)

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
    delta_a   = d['delta_a'][::D].astype(DTYPE_NP)
    vdelta_a  = np.zeros_like(delta_a)
    vdelta_a[1:] = (delta_a[1:] - delta_a[:-1]) * FS_NEW
    vdelta_a[0]  = vdelta_a[1]
    return x_logical, np.stack([delta_a, vdelta_a], axis=1)

train_list  = [load_traj(f) for f in TRAIN_FILES]
train_data  = deepSI.System_data_list(train_list)
val_data    = load_traj(VAL_FILE)
val_x_logical, val_x_aug = load_mat_aug(VAL_FILE)
print(f'Loaded data. Val: {val_data.y.shape}  x_logical: {val_x_logical.shape}')

## ==============================================================================═
## Normalisation — copied verbatim
## ==============================================================================═

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

## ==============================================================================═
## build_model — copied verbatim from gantry_interconnect_dynamic.py
## ==============================================================================═

def _get_encoder_dims(hp):
    na = 4 * NX_PHYS + 1; nb = na; na_right = 1; nb_right = 1
    return na, nb, na_right, nb_right

def build_model(hp):
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    na, nb, na_right, nb_right = _get_encoder_dims(hp)
    ic = Interconnect(nxd, nu, ny, debugging=False)
    phy_block = Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u,
                                   x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
                                   up_sample=hp['up_sample']).to(DTYPE_PT)
    C_aug_init = np.zeros((ny, NX_ANN), dtype=DTYPE_NP)
    C_aug_init[2, 0] = 1e-2
    out_phys = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    out_aug  = Parameterized_Linear_Output_Block(C=C_aug_init,
                D=np.zeros((ny, nu), dtype=DTYPE_NP), flag_loss_reg=False)
    AUG_IX = np.arange(NX_PHYS, nxd)
    ann_block = Static_ANN_Block(nz=nxd + nu, nw=NX_ANN,
                                 n_nodes_per_layer=hp['n_nodes_per_layer'],
                                 n_hidden_layers=hp['n_hidden_layers'],
                                 net=zero_init_feed_forward_nn,
                                 activation=torch.nn.Tanh)
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
                               na_right=na_right, nb_right=nb_right,
                               e_net_kwargs={"n_nodes_per_layer": hp['n_nodes_per_layer'],
                                             "n_hidden_layers": hp['n_hidden_layers']})
    fit_sys.norm.u0 = u_mean.flatten(); fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0 = y0;              fit_sys.norm.ystd = ystd

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

    fit_sys.encoder = linear_encoder_init_aug(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
        nx_aug=hp['NX_ANN'],
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        flag_linear_only=False,
        u_mean=u_mean, std_u=std_u, y0=y0, ystd=ystd,
        x_mean=x_mean, std_x=std_x,
    ).to(DTYPE_PT)
    fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
    fit_sys.hfn.to(DTYPE_PT)
    return fit_sys, Ad_bar, Bd_bar, Cd_bar, Dd_bar

## ==============================================================================═
## Build model
## ==============================================================================═

print('\nBuilding model (no training)...')
fit_sys, Ad_bar, Bd_bar, Cd_bar, Dd_bar = build_model(DEFAULT_HP)
fit_sys.eval()
NX_ANN = DEFAULT_HP['NX_ANN']
nxd    = NX_PHYS + NX_ANN
na, nb, na_right, nb_right = _get_encoder_dims(DEFAULT_HP)

labels_phys = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']
t_val = np.arange(len(val_data.y)) * val_data.dt

## ==============================================================================═
## Part 1 — Observability analysis
## ==============================================================================═

print('\n' + '='*60)
print('Part 1: Observability / reconstructability analysis')
print('='*60)

n = na  # window length
A, C = Ad_bar, Cd_bar

# Observability matrix O_n: shape ((n+1)*ny, nx)
O_n = np.zeros(((n + 1) * ny, NX_PHYS), dtype=np.float64)
for i in range(n + 1):
    O_n[i * ny:(i + 1) * ny, :] = C @ np.linalg.matrix_power(A, i)

U_svd, S_svd, Vt_svd = np.linalg.svd(O_n, full_matrices=False)
O_pinv = np.linalg.pinv(O_n)  # shape (nx, (n+1)*ny)

# Per-channel noise amplification: ||row j of O_pinv||
# Larger = more sensitive to noise = harder to reconstruct
noise_amp = np.linalg.norm(O_pinv, axis=1)  # (nx,)

print(f'\nObservability matrix O_n: shape {O_n.shape}')
print(f'Singular values (top 10): {S_svd[:10].round(4)}')
print(f'Condition number: {S_svd[0] / S_svd[-1]:.2f}')
print('\nPer-channel noise amplification ||pinv(O_n)[j,:]||:')
for ch in range(NX_PHYS):
    print(f'  {labels_phys[ch]:5s}: {noise_amp[ch]:.4f}')

# Plot
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
axes1[0].semilogy(S_svd, 'o-', ms=4)
axes1[0].set_xlabel('Singular value index'); axes1[0].set_ylabel('Singular value')
axes1[0].set_title(f'O_n singular values (cond={S_svd[0]/S_svd[-1]:.1f})')
axes1[0].grid(True, which='both')

colors = ['C0' if 'q' in l and 'd' not in l else 'C1' for l in labels_phys]
bars = axes1[1].bar(labels_phys, noise_amp, color=colors)
axes1[1].set_ylabel('||pinv(O_n)[j, :]||  (noise amplification)')
axes1[1].set_title('Per-channel reconstruction sensitivity\nblue=position, orange=velocity')
axes1[1].grid(True, axis='y')
for bar, val in zip(bars, noise_amp):
    axes1[1].text(bar.get_x() + bar.get_width()/2, val, f'{val:.2f}',
                  ha='center', va='bottom', fontsize=8)
fig1.suptitle('diag19 — Observability analysis (n=25 samples at 4000 Hz)')
fig1.tight_layout()
fig1.savefig(os.path.join(SAVE_DIR, f'{RID}_observability.png'), dpi=150)
print(f'\nSaved: {RID}_observability.png')

## ==============================================================================═
## Part 2 — Encoder bias analysis
## ==============================================================================═

print('\n' + '='*60)
print('Part 2: Encoder residual — bias or noise?')
print('='*60)

x_true_norm = (val_x_logical.astype(DTYPE_NP) - x_mean.flatten()) / std_x.flatten()

val_norm = fit_sys.norm.transform(val_data)
yn = np.ascontiguousarray(val_norm.y, dtype=DTYPE_NP)
un = np.ascontiguousarray(val_norm.u, dtype=DTYPE_NP)
N  = len(yn)
k0 = max(na, nb) + 1
stride = max(1, (N - k0) // 2000)
k_ix = np.arange(k0, N, stride)
ypast = np.stack([yn[k - na : k + na_right] for k in k_ix])
upast = np.stack([un[k - nb : k + nb_right] for k in k_ix])
with torch.no_grad():
    x_hat = fit_sys.encoder(
        torch.tensor(upast, dtype=DTYPE_PT),
        torch.tensor(ypast, dtype=DTYPE_PT),
    ).numpy()  # (Nk, nxd)

x_hat_phys = x_hat[:, :NX_PHYS]
x_true_k   = x_true_norm[k_ix]
residual    = x_hat_phys - x_true_k  # (Nk, NX_PHYS)
t_k         = t_val[k_ix]

print('\nEncoder residual statistics (normalized units):')
print(f'  {"channel":6s}  {"mean (bias)":>14s}  {"std (noise)":>14s}  {"max|resid|":>12s}')
for ch in range(NX_PHYS):
    r = residual[:, ch]
    print(f'  {labels_phys[ch]:6s}  {r.mean():+14.4f}  {r.std():14.4f}  {np.abs(r).max():12.4f}')

fig2, axes2 = plt.subplots(NX_PHYS, 1, figsize=(12, 10), sharex=True)
for ch, ax in enumerate(axes2):
    r = residual[:, ch]
    ax.plot(t_k, r, lw=0.6, color='C0', alpha=0.8)
    ax.axhline(r.mean(), color='C1', lw=1.5, linestyle='--',
               label=f'mean={r.mean():+.4f}  std={r.std():.4f}')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel(f'Δ{labels_phys[ch]}\n(norm)'); ax.legend(fontsize=7, loc='upper right')
    ax.grid(True)
axes2[-1].set_xlabel('Time [s]')
fig2.suptitle('diag19 — Encoder residual per channel (x_hat - x_true, normalized)\n'
              'Flat mean near zero = noise; Drifting mean = systematic bias')
fig2.tight_layout()
fig2.savefig(os.path.join(SAVE_DIR, f'{RID}_encoder_bias.png'), dpi=150)
print(f'Saved: {RID}_encoder_bias.png')

## ==============================================================================═
## Part 3 — Oracle-init HFN simulation
## ==============================================================================═

print('\n' + '='*60)
print('Part 3: Oracle-init HFN simulation (true x_logical, x_aug=0)')
print('='*60)

val_norm2    = fit_sys.norm.transform(val_data)
u_val_norm   = torch.tensor(np.ascontiguousarray(val_norm2.u), dtype=DTYPE_PT)
y_ref        = val_data.y

# Oracle init: true x_logical[0] normalized, x_aug = 0
x0_phys_norm = (val_x_logical[0].astype(DTYPE_NP) - x_mean.flatten()) / std_x.flatten()
x_oracle     = torch.zeros(1, nxd, dtype=DTYPE_PT)
x_oracle[0, :NX_PHYS] = torch.tensor(x0_phys_norm, dtype=DTYPE_PT)

y_oracle_list = []
with torch.no_grad():
    for t in range(len(u_val_norm)):
        y_t, x_oracle = fit_sys.hfn(x_oracle, u_val_norm[t:t+1])
        y_oracle_list.append(y_t.squeeze().numpy())
y_hat_oracle = np.array(y_oracle_list, dtype=DTYPE_NP) * ystd + y0

nrms_oracle = np.sqrt(((y_hat_oracle - y_ref)**2).mean(axis=0)) / ystd
rms_oracle  = nrms_oracle * ystd

print('\n=== Oracle-init HFN sim-NRMS (x_logical init, x_aug=0, no training) ===')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    print(f'  {lbl}: {nrms_oracle[ch]:.4f}  ({rms_oracle[ch]*1e6:.1f} um)')

# Baseline FP (oracle, manual)
phy_block_fp = Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u,
                                   x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
                                   up_sample=DEFAULT_HP['up_sample']).to(DTYPE_PT)
phy_block_fp.eval()
x_norm_fp = (val_x_logical[0].astype(DTYPE_NP) - x_mean.flatten()) / std_x.flatten()
u_val_norm_np = ((val_data.u - u_mean.flatten()) / std_u.flatten()).astype(DTYPE_NP)
y_fp_list = []
with torch.no_grad():
    for t in range(len(val_data.u)):
        y_norm = Cd_norm @ x_norm_fp + Dd_np @ u_val_norm_np[t]
        y_fp_list.append(y_norm * ystd + y0)
        z = torch.cat([torch.tensor(x_norm_fp, dtype=DTYPE_PT).view(1, NX_PHYS, 1),
                       torch.tensor(u_val_norm_np[t], dtype=DTYPE_PT).view(1, nu, 1)], dim=1)
        x_norm_fp = phy_block_fp(z).view(NX_PHYS).cpu().numpy()
y_hat_fp  = np.array(y_fp_list, dtype=DTYPE_NP)
nrms_fp   = np.sqrt(((y_hat_fp - y_ref)**2).mean(axis=0)) / ystd
rms_fp    = nrms_fp * ystd

print('\n=== Baseline FP sim-NRMS (oracle init, no augmentation) ===')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    print(f'  {lbl}: {nrms_fp[ch]:.4f}  ({rms_fp[ch]*1e6:.1f} um)')

# Encoder-init sim (from diag18, reproduce here)
print('\nRunning encoder-init simulation...')
sim_result   = fit_sys.apply_experiment(val_data)
cheat_n      = sim_result.cheat_n
y_hat_enc    = sim_result.y
nrms_enc     = np.sqrt(((y_hat_enc[cheat_n:] - y_ref[cheat_n:])**2).mean(axis=0)) / ystd
rms_enc      = nrms_enc * ystd

print('\n=== Encoder-init sim-NRMS (W^b init, no training) ===')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    print(f'  {lbl}: {nrms_enc[ch]:.4f}  ({rms_enc[ch]*1e6:.1f} um)')

print('\n=== Summary: diagnosing the 12x gap ===')
print(f'  {"channel":4s}  {"baseline FP":>14s}  {"oracle HFN":>14s}  {"encoder-init":>14s}')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    print(f'  {lbl}   '
          f'{nrms_fp[ch]:.4f} ({rms_fp[ch]*1e6:.0f}um)  '
          f'{nrms_oracle[ch]:.4f} ({rms_oracle[ch]*1e6:.0f}um)  '
          f'{nrms_enc[ch]:.4f} ({rms_enc[ch]*1e6:.0f}um)')
print('\nInterpretation:')
print('  baseline FP ~= oracle HFN  -> HFN model correct at init')
print('  oracle HFN  ~= encoder-init -> problem is HFN model, not encoder')
print('  oracle HFN  << encoder-init -> problem is encoder state estimation')

## ==============================================================================═
## Plot — NRMS comparison bar chart
## ==============================================================================═

ch_labels = ['X1', 'X2', 'Y']
x_pos  = np.arange(ny)
width  = 0.25

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.bar(x_pos - width, nrms_fp,     width, label='Baseline FP (oracle)',    color='C2', alpha=0.85)
ax3.bar(x_pos,         nrms_oracle, width, label='Oracle-init HFN (x_aug=0)',color='C1', alpha=0.85)
ax3.bar(x_pos + width, nrms_enc,    width, label='Encoder-init HFN (W^b)',  color='C0', alpha=0.85)
ax3.set_xticks(x_pos); ax3.set_xticklabels(ch_labels)
ax3.set_ylabel('sim-NRMS')
ax3.set_title('diag19 — NRMS comparison: isolating encoder vs HFN model error\n'
              'If oracle HFN ≈ baseline FP: HFN model is correct, encoder is the bottleneck')
ax3.legend(fontsize=9); ax3.grid(True, axis='y')
fig3.tight_layout()
fig3.savefig(os.path.join(SAVE_DIR, f'{RID}_nrms_comparison.png'), dpi=150)
print(f'\nSaved: {RID}_nrms_comparison.png')

plt.close('all')

np.savez(os.path.join(SAVE_DIR, f'{RID}_results.npz'),
         noise_amp=noise_amp, S_svd=S_svd, O_n=O_n,
         residual=residual, k_ix=k_ix, t_k=t_k,
         nrms_fp=nrms_fp, nrms_oracle=nrms_oracle, nrms_enc=nrms_enc,
         rms_fp=rms_fp, rms_oracle=rms_oracle, rms_enc=rms_enc,
         labels_phys=np.array(labels_phys))
print(f'Saved: {RID}_results.npz')

print('\n' + '='*60)
print('diag19 complete')
print('='*60)
