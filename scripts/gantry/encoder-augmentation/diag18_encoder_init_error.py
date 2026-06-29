"""diag18_encoder_init_error.py

Measures encoder initialization quality BEFORE any training.

Answers:
  - How well does linear_encoder_init_aug reproduce x_phys at t=0? (W^b quality)
  - How correlated is x_aug with delta_a/vdelta_a at t=0? (W^a quality)
  - What is the initial sim-NRMS before any gradient step?

Copies setup from gantry_interconnect_dynamic.py exactly.
No training is run.

Outputs (to diagnostics/):
  - Console: initial sim-NRMS, R2_raw + R2_linmap for all channels
  - diag18_state_recovery_init.npz
  - diag18_xphys_init.png  -- encoder x_phys estimate vs x_logical (GT)
  - diag18_xaug_init.png   -- encoder x_aug vs delta_a/vdelta_a (GT)
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model_augmentation.utils.utils import *
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn, HybridGantryEncoder
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
from model_augmentation.utils.utils import normalize_linear_ss_matrices

## ═══════════════════════════════════════════════════════════════════════════════
## Configuration — copied verbatim from gantry_interconnect_dynamic.py
## ═══════════════════════════════════════════════════════════════════════════════

MODE = 'multisine'
NX_PHYS = 6
nu  = 3
ny  = 3
Y_OP = None
ENCODER_INIT = 'linear_map'
USE_HYBRID_ENCODER = (ENCODER_INIT == 'hybrid')
ANN_ACTIVATION = 'tanh'
SEED = 42

FS_ORIG = 20000
FS_NEW  = 4000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

USE_F64  = False
DTYPE_NP = np.float64    if USE_F64 else np.float32
DTYPE_PT = torch.float64 if USE_F64 else torch.float32

save_flag = True

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

## ═══════════════════════════════════════════════════════════════════════════════
## Data loading — copied verbatim
## ═══════════════════════════════════════════════════════════════════════════════

np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_SUBDIR = 'multisine' if MODE == 'multisine' else 'trajectories'
TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry', 'matlab', DATA_SUBDIR)
print(f'Data dir ({MODE}): {TRAJ_DIR}')

TRAIN_FILES = [
    'T1_Y_sweep_conservative.mat',
    'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat',
    'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat',
    'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat',
    'T8_X_sym_anti_Y_sweep.mat',
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

def load_mat_aug(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u         = _load_u(d)[::D].astype(DTYPE_NP)
    y         = d['y'][::D].astype(DTYPE_NP)
    x_logical = d['x_logical'][::D].astype(DTYPE_NP)
    delta_a   = d['delta_a'][::D].astype(DTYPE_NP)
    vdelta_a      = np.zeros_like(delta_a)
    vdelta_a[1:]  = (delta_a[1:] - delta_a[:-1]) * FS_NEW
    vdelta_a[0]   = vdelta_a[1]
    x_aug = np.stack([delta_a, vdelta_a], axis=1)
    return u, y, x_logical, x_aug

train_list = [load_traj(f) for f in TRAIN_FILES]
train_data = deepSI.System_data_list(train_list)
val_data   = load_traj(VAL_FILE)

_, _, val_x_logical, val_x_aug = load_mat_aug(VAL_FILE)
print(f'Loaded val augmented GT: x_logical={val_x_logical.shape}  '
      f'delta_a std={val_x_aug[:,0].std():.3e} m  '
      f'vdelta_a std={val_x_aug[:,1].std():.3e} m/s')

print(f'Loaded {len(train_list)} training trajectories, 1 val')

## ═══════════════════════════════════════════════════════════════════════════════
## Normalisation — copied verbatim
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
y0     = y_all.mean(axis=0).astype(DTYPE_NP)

Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
Dd_np   = Dd.numpy()

PHY_IX = np.arange(NX_PHYS)

## ═══════════════════════════════════════════════════════════════════════════════
## Functions — copied verbatim from gantry_interconnect_dynamic.py
## ═══════════════════════════════════════════════════════════════════════════════

def _get_encoder_dims(hp):  # mirrors fix in gantry_interconnect_dynamic.py (na=17 via Jan's rule)
    if ENCODER_INIT == 'linear_map':
        na = hp.get('na_nb', 2 * (NX_PHYS + hp['NX_ANN']) + 1)  # THEORY: nxd*2+1 (Jan's standard)
        nb = na
        na_right = 1   # reconstructability map uses y(k), window is [k-na, k]
        nb_right = 1
    else:
        na = hp.get('na_nb', 2 * (NX_PHYS + hp['NX_ANN']) + 1)
        nb = na
        na_right = 0
        nb_right = 0
    return na, nb, na_right, nb_right


def build_model(hp):
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN

    na, nb, na_right, nb_right = _get_encoder_dims(hp)

    ic = Interconnect(nxd, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
        up_sample=hp['up_sample'],
    ).to(DTYPE_PT)

    C_aug_init = np.zeros((ny, NX_ANN), dtype=DTYPE_NP)
    C_aug_init[2, 0] = 1e-2   # HEURISTIC: Y <- delta_a
    out_phys = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    out_aug  = Parameterized_Linear_Output_Block(
        C=C_aug_init, D=np.zeros((ny, nu), dtype=DTYPE_NP), flag_loss_reg=False)
    ic.add_block(phy_block)
    ic.add_block(out_phys)
    ic.add_block(out_aug)

    _act = torch.nn.Identity if ANN_ACTIVATION == 'linear' else torch.nn.Tanh
    AUG_IX = np.arange(NX_PHYS, nxd)
    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=NX_ANN,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=_act,
    )
    ic.add_block(ann_block)

    ic.connect_block_signals(ann_block, ["x", "u"], [])
    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(AUG_IX, nxd))
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", out_phys, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_phys, ["u"], ["y"])
    ic.connect_signals("x", out_aug, "concat", selection_matrix(AUG_IX, nxd))
    ic.connect_block_signals(out_aug, ["u"], ["y"])

    fit_sys = SSE_Interconnect(
        interconnect=ic, na=na, nb=nb,
        na_right=na_right, nb_right=nb_right,
        e_net_kwargs={
            "n_nodes_per_layer": hp['n_nodes_per_layer'],
            "n_hidden_layers": hp['n_hidden_layers'],
        },
    )

    fit_sys.norm.u0   = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0   = y0
    fit_sys.norm.ystd = ystd

    if ENCODER_INIT == 'linear_map':
        Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)

        baseline_npz_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry',
            'baseline_simulations', f'{MODE}_LPV', 'baseline_states.npz')
        if os.path.exists(baseline_npz_path):
            bl = np.load(baseline_npz_path, allow_pickle=True)
            x_phys_all = np.concatenate(bl['x_train_phys'])
        else:
            x_phys_all = x_all
            print("WARNING: baseline_states.npz not found, using finite-diff states for normalization")

        sys_data_with_x = deepSI.System_data(u=u_all, y=y_all)
        sys_data_with_x.x = x_phys_all

        Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
            Ad, Bd, Cd_dt, Dd_dt, sys_data_with_x)

        fit_sys.encoder = linear_encoder_init_aug(
            A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
            nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
            nx_aug=DEFAULT_HP['NX_ANN'],
            n_nodes_per_layer=hp['n_nodes_per_layer'],
            n_hidden_layers=hp['n_hidden_layers'],
            flag_linear_only=False,
            u_mean=u_mean, std_u=std_u,
            y0=y0, ystd=ystd,
            x_mean=x_mean, std_x=std_x,
        ).to(DTYPE_PT)

    fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
    if ENCODER_INIT in ('hybrid', 'linear_map'):
        fit_sys.hfn.to(DTYPE_PT)
    else:
        for net in (fit_sys.encoder, fit_sys.hfn):
            net.to(DTYPE_PT)

    return fit_sys


def aug_state_r2(fit_sys, hp):
    NX_ANN = hp['NX_ANN']
    na, nb, na_right, nb_right = _get_encoder_dims(hp)
    fit_sys.eval()

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
        ).numpy()

    x_ann = x_hat[:, NX_PHYS:]

    gt_raw  = val_x_aug[k_ix]
    gt_mean = gt_raw.mean(axis=0)
    gt_std  = gt_raw.std(axis=0) + 1e-8
    gt_norm = (gt_raw - gt_mean) / gt_std

    def _r2(ref, est):
        ss_res = ((ref - est)**2).sum(axis=0)
        ss_tot = ((ref - ref.mean(axis=0))**2).sum(axis=0)
        return 1.0 - ss_res / (ss_tot + 1e-12)

    r2_raw = _r2(gt_norm, x_ann)

    A_aug = np.hstack([x_ann, np.ones((len(x_ann), 1), dtype=DTYPE_NP)])
    W_aug, *_ = np.linalg.lstsq(A_aug, gt_norm, rcond=None)
    r2_lin = _r2(gt_norm, A_aug @ W_aug)

    return r2_raw, r2_lin


def state_recovery_diagnostic(fit_sys, hp, rid, max_windows=2000):
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    na, nb, na_right, nb_right = _get_encoder_dims(hp)
    fit_sys.eval()

    x_true = val_x_logical.astype(DTYPE_NP)
    x_true_norm = (x_true - x_mean.flatten()) / std_x.flatten()

    val_norm = fit_sys.norm.transform(val_data)
    yn = np.ascontiguousarray(val_norm.y, dtype=DTYPE_NP)
    un = np.ascontiguousarray(val_norm.u, dtype=DTYPE_NP)
    N = len(yn)
    k0 = max(na, nb) + 1
    stride = max(1, (N - k0) // max_windows)
    k_ix = np.arange(k0, N, stride)
    ypast = np.stack([yn[k - na : k + na_right] for k in k_ix])
    upast = np.stack([un[k - nb : k + nb_right] for k in k_ix])
    with torch.no_grad():
        x_hat = fit_sys.encoder(
            torch.tensor(upast, dtype=DTYPE_PT),
            torch.tensor(ypast, dtype=DTYPE_PT),
        ).numpy()

    xt   = x_true_norm[k_ix]
    xt_l = x_true_norm[k_ix - 1]

    def r2_per_channel(ref, est):
        ss_res = ((ref - est) ** 2).sum(axis=0)
        ss_tot = ((ref - ref.mean(axis=0)) ** 2).sum(axis=0)
        return 1.0 - ss_res / ss_tot

    r2_raw = r2_per_channel(xt,   x_hat[:, :NX_PHYS])
    r2_lag = r2_per_channel(xt_l, x_hat[:, :NX_PHYS])

    A = np.hstack([x_hat, np.ones((len(x_hat), 1), dtype=DTYPE_NP)])
    W, *_ = np.linalg.lstsq(A, xt, rcond=None)
    r2_lin = r2_per_channel(xt, A @ W)

    labels = ['q1 ', 'q2 ', 'q3 ', 'dq1', 'dq2', 'dq3']
    print('\n=== State recovery diagnostic (init, before training) ===')
    print(f'  {len(k_ix)} windows (stride {stride}), na=nb={na}')
    print('  channel   R2_raw      R2_linmap   R2_raw_lag1')
    for ch in range(NX_PHYS):
        print(f'  {labels[ch]}     {r2_raw[ch]:+10.4f}  {r2_lin[ch]:+10.4f}  {r2_lag[ch]:+10.4f}')
    print('  R2_linmap ~ 1 & R2_raw low -> basis rotation;')
    print('  R2_linmap low              -> information absent from encoder state;')
    print('  R2_raw_lag1 > R2_raw       -> encoder aligned to k-1 (one-sample lag)')

    r2_aug_raw, r2_aug_lin = aug_state_r2(fit_sys, hp)
    aug_labels = ['delta_a  ', 'vdelta_a ']
    aug_notes  = ['(mat file)', '(FD estimate)']
    print('\n=== Augmented state R2 vs GT (init, before training) ===')
    print(f'  {"state":<12s}  {"R2_raw":>10s}  {"R2_linmap":>10s}  note')
    for ch in range(hp['NX_ANN']):
        lbl  = aug_labels[ch] if ch < len(aug_labels) else f'x_ann[{ch}]'
        note = aug_notes[ch]  if ch < len(aug_notes)  else ''
        print(f'  {lbl}  {r2_aug_raw[ch]:+10.4f}  {r2_aug_lin[ch]:+10.4f}  {note}')

    if save_flag:
        np.savez(os.path.join(SAVE_DIR, f'{rid}_state_recovery.npz'),
                 r2_raw=r2_raw, r2_lin=r2_lin, r2_lag=r2_lag,
                 W=W, k_ix=k_ix, x_hat=x_hat, x_true_norm=xt,
                 r2_aug_raw=r2_aug_raw, r2_aug_lin=r2_aug_lin)
        print(f'Saved: {rid}_state_recovery.npz')

    return x_hat, k_ix, r2_raw, r2_lin, r2_lag, r2_aug_raw, r2_aug_lin


def compute_baseline_fp_nrms(hp):
    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
        up_sample=hp['up_sample'],
    ).to(DTYPE_PT)
    phy_block.eval()

    x0_phys   = val_x_logical[0].astype(DTYPE_NP)
    x_norm_np = (x0_phys - x_mean.flatten()) / std_x.flatten()

    u_val_norm = ((val_data.u - u_mean.flatten()) / std_u.flatten()).astype(DTYPE_NP)

    y_hat_list = []
    with torch.no_grad():
        for t in range(len(val_data.u)):
            u_norm_np = u_val_norm[t]
            y_norm = Cd_norm @ x_norm_np + Dd_np @ u_norm_np
            y_phys = y_norm * ystd + y0
            y_hat_list.append(y_phys)
            x_t = torch.tensor(x_norm_np, dtype=DTYPE_PT).view(1, NX_PHYS, 1)
            u_t = torch.tensor(u_norm_np, dtype=DTYPE_PT).view(1, nu, 1)
            z   = torch.cat([x_t, u_t], dim=1)
            x_norm_next = phy_block(z)
            x_norm_np = x_norm_next.view(NX_PHYS).cpu().numpy()

    y_hat = np.array(y_hat_list, dtype=DTYPE_NP)
    y_ref  = val_data.y
    nrms   = np.sqrt(((y_hat - y_ref)**2).mean(axis=0)) / ystd
    rms    = nrms * ystd

    print('\n=== Baseline FP model sim-NRMS (no MSD, oracle init) ===')
    for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
        print(f'  {lbl}: {nrms[ch]:.4f}  ({rms[ch]*1e6:.1f} µm)')
    return nrms, y_hat


## ═══════════════════════════════════════════════════════════════════════════════
## Main — init quality measurement only, no training
## ═══════════════════════════════════════════════════════════════════════════════

RID = 'diag18'

print('\n' + '='*60)
print('diag18 — encoder initialization error (no training)')
print('='*60)

np.random.seed(SEED)
torch.manual_seed(SEED)
fit_sys = build_model(DEFAULT_HP)
fit_sys.eval()

# ── Initial simulation (encoder-initialized, no training) ───────────────────
print('\nRunning initial simulation (encoder-init, 0 epochs)...')
sim_result = fit_sys.apply_experiment(val_data)
cheat_n    = sim_result.cheat_n
y_hat_init = sim_result.y
y_ref      = val_data.y
t_val      = np.arange(len(y_ref)) * val_data.dt
cheat_t    = cheat_n * val_data.dt

nrms_init = np.sqrt(((y_hat_init[cheat_n:] - y_ref[cheat_n:])**2).mean(axis=0)) / ystd
rms_init  = nrms_init * ystd

print('\n=== Initial sim-NRMS (encoder-init, 0 epochs) ===')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    print(f'  {lbl}: {nrms_init[ch]:.4f}  ({rms_init[ch]*1e6:.1f} µm)')

# ── Baseline FP for reference ────────────────────────────────────────────────
baseline_nrms, y_hat_fp = compute_baseline_fp_nrms(DEFAULT_HP)

print('\n=== Init vs baseline comparison ===')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    delta = 100.0 * (baseline_nrms[ch] - nrms_init[ch]) / (baseline_nrms[ch] + 1e-12)
    print(f'  {lbl}: init={nrms_init[ch]:.4f}  baseline_FP={baseline_nrms[ch]:.4f}  diff={delta:+.1f}%')

# ── State recovery at init ───────────────────────────────────────────────────
x_hat, k_ix, r2_raw, r2_lin, r2_lag, r2_aug_raw, r2_aug_lin = \
    state_recovery_diagnostic(fit_sys, DEFAULT_HP, RID)

NX_ANN = DEFAULT_HP['NX_ANN']
nxd    = NX_PHYS + NX_ANN
na, nb, na_right, nb_right = _get_encoder_dims(DEFAULT_HP)

# ── Plot 1: x_phys encoder estimate vs GT (normalized) ──────────────────────
x_true_norm = (val_x_logical.astype(DTYPE_NP) - x_mean.flatten()) / std_x.flatten()
labels_phys  = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']
t_k = t_val[k_ix]

fig1, axes1 = plt.subplots(NX_PHYS, 1, figsize=(12, 10), sharex=True)
for ch, ax in enumerate(axes1):
    ax.plot(t_k, x_true_norm[k_ix, ch], 'k', lw=0.8, label='GT x_logical (norm)')
    ax.plot(t_k, x_hat[:, ch], 'C0', lw=0.8, alpha=0.8,
            label=f'Encoder init  R2_raw={r2_raw[ch]:+.3f}  R2_lin={r2_lin[ch]:+.3f}')
    ax.set_ylabel(labels_phys[ch]); ax.legend(fontsize=6, loc='upper right'); ax.grid(True)
axes1[-1].set_xlabel('Time [s]')
fig1.suptitle('diag18 — Physical state encoder at init (before training)\nblue=encoder, black=GT')
fig1.tight_layout()
fig1.savefig(os.path.join(SAVE_DIR, f'{RID}_xphys_init.png'), dpi=150)
print(f'\nSaved: {RID}_xphys_init.png')

# ── Plot 2: x_aug encoder estimate vs GT ────────────────────────────────────
aug_gt_labels = ['delta_a [m]', 'vdelta_a [m/s]']
aug_gt_names  = ['delta_a', 'vdelta_a']

fig2, axes2 = plt.subplots(NX_ANN, 1, figsize=(12, 4), sharex=True)
if NX_ANN == 1:
    axes2 = [axes2]
for ch, ax in enumerate(axes2):
    ax.plot(t_k, x_hat[:, NX_PHYS + ch], 'C0', lw=0.8,
            label=f'x_ann[{NX_PHYS+ch}] (encoder init)  '
                  f'R2_raw={r2_aug_raw[ch]:+.4f}  R2_lin={r2_aug_lin[ch]:+.4f}')
    ax.set_ylabel(f'x[{NX_PHYS+ch}] (dim-less)', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    if ch < len(aug_gt_labels):
        ax2 = ax.twinx()
        ax2.plot(t_k, val_x_aug[k_ix, ch], 'C1', lw=0.8, alpha=0.7,
                 label=f'GT {aug_gt_names[ch]}')
        ax2.set_ylabel(aug_gt_labels[ch], color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=6, loc='upper right')
    else:
        ax.legend(fontsize=6)
    ax.grid(True)
axes2[-1].set_xlabel('Time [s]')
fig2.suptitle('diag18 — Augmented state encoder at init (before training)\nblue=encoder, orange=GT')
fig2.tight_layout()
fig2.savefig(os.path.join(SAVE_DIR, f'{RID}_xaug_init.png'), dpi=150)
print(f'Saved: {RID}_xaug_init.png')

plt.close('all')

if save_flag:
    np.savez(
        os.path.join(SAVE_DIR, f'{RID}_results.npz'),
        nrms_init=nrms_init, rms_init=rms_init,
        baseline_nrms=baseline_nrms,
        y_hat_init=y_hat_init, y_ref=y_ref, t_val=t_val,
        x_hat=x_hat, k_ix=k_ix,
        x_true_norm=x_true_norm,
        val_x_aug=val_x_aug,
        r2_raw=r2_raw, r2_lin=r2_lin, r2_lag=r2_lag,
        r2_aug_raw=r2_aug_raw, r2_aug_lin=r2_aug_lin,
    )
    print(f'Saved: {RID}_results.npz')

print('\n' + '='*60)
print('diag18 complete')
print('='*60)
