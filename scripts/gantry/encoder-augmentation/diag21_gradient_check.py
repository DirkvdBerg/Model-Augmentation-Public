"""diag21_gradient_check.py

Quick check: is the model learnable from the na=17 initialization?

Steps:
  1. Build model (DEFAULT_HP, na=17 via Jan's rule nxd*2+1)
  2. Compute init sim-NRMS (0 gradient steps)
  3. Run N_CHECK_EPOCHS training epochs via fit_sys.fit() directly
     (no DIAG_INTERVAL chunking -- avoids the DIAG_INTERVAL bug)
  4. Recompute sim-NRMS after each epoch
  5. Print before/after comparison vs baseline FP

Expected: NRMS should drop below init (0.0041/0.0037/0.0040) within 1-2 epochs
if the gradient is well-conditioned from the na=17 init.

No files saved. Console output only. Runtime ~1-2 min.
"""

import os
import sys
import numpy as np
import torch
import deepSI
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model_augmentation.utils.utils import *
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
from model_augmentation.utils.utils import normalize_linear_ss_matrices

## ============================================================
## Configuration — verbatim from gantry_interconnect_dynamic.py
## ============================================================

MODE         = 'multisine'
NX_PHYS      = 6
nu           = 3
ny           = 3
Y_OP         = None
ENCODER_INIT = 'linear_map'
ANN_ACTIVATION = 'tanh'
SEED         = 42

FS_ORIG = 20000
FS_NEW  = 4000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

USE_F64  = False
DTYPE_NP = np.float64    if USE_F64 else np.float32
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
    epochs=10,
)
DEFAULT_HP['na_nb'] = (NX_PHYS + DEFAULT_HP['NX_ANN']) * 2 + 1  # THEORY: nxd*2+1 (Jan's standard) = 17

# How many training epochs to run in this diagnostic
N_CHECK_EPOCHS = 2

## ============================================================
## Data loading — verbatim
## ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_SUBDIR = 'multisine'
TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry', 'matlab', DATA_SUBDIR)

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
VAL_FILE = 'V1_X_sym_Y_mid_sweep.mat'

def _load_u(d):
    return d['u_total'] if 'u_total' in d else d['u']

def load_traj(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    return deepSI.System_data(
        u=_load_u(d)[::D].astype(DTYPE_NP),
        y=d['y'][::D].astype(DTYPE_NP),
        dt=TS_NEW,
    )

def load_mat_aug(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = _load_u(d)[::D].astype(DTYPE_NP)
    y = d['y'][::D].astype(DTYPE_NP)
    x_logical = d['x_logical'][::D].astype(DTYPE_NP)
    delta_a   = d['delta_a'][::D].astype(DTYPE_NP)
    # HEURISTIC: backward FD for velocity -- O(Ts) accurate
    vdelta_a      = np.zeros_like(delta_a)
    vdelta_a[1:]  = (delta_a[1:] - delta_a[:-1]) * FS_NEW
    vdelta_a[0]   = vdelta_a[1]
    x_aug = np.stack([delta_a, vdelta_a], axis=1)
    return u, y, x_logical, x_aug

train_list = [load_traj(f) for f in TRAIN_FILES]
train_data = deepSI.System_data_list(train_list)
val_data   = load_traj(VAL_FILE)
_, _, val_x_logical, val_x_aug = load_mat_aug(VAL_FILE)

print(f'Loaded {len(train_list)} training trajectories, 1 val')
print(f'na_nb = {DEFAULT_HP["na_nb"]}  ({DEFAULT_HP["na_nb"]/FS_NEW*1000:.2f} ms window)')

## ============================================================
## Normalisation — verbatim
## ============================================================

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

## ============================================================
## Model building — verbatim from diag18 (na=17 fix applied)
## ============================================================

def _get_encoder_dims(hp):
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
            print("WARNING: baseline_states.npz not found, using finite-diff states")

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


## ============================================================
## Helpers
## ============================================================

def sim_nrms(fit_sys):
    """Run apply_experiment on val_data, return per-channel NRMS (ny,)."""
    fit_sys.eval()
    sim_result = fit_sys.apply_experiment(val_data)
    cheat_n = sim_result.cheat_n
    y_hat   = sim_result.y          # (T, ny) physical [m]
    y_ref   = val_data.y
    return np.sqrt(((y_hat[cheat_n:] - y_ref[cheat_n:]) ** 2).mean(axis=0)) / ystd


def compute_baseline_fp_nrms():
    """Baseline FP sim-NRMS (no MSD, oracle init)."""
    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
        up_sample=DEFAULT_HP['up_sample'],
    ).to(DTYPE_PT)
    phy_block.eval()

    x0_phys   = val_x_logical[0].astype(DTYPE_NP)
    x_norm_np = (x0_phys - x_mean.flatten()) / std_x.flatten()
    u_val_norm = ((val_data.u - u_mean.flatten()) / std_u.flatten()).astype(DTYPE_NP)

    y_hat_list = []
    with torch.no_grad():
        for t_step in range(len(val_data.u)):
            u_norm_np = u_val_norm[t_step]
            y_norm = Cd_norm @ x_norm_np + Dd_np @ u_norm_np
            y_hat_list.append(y_norm * ystd + y0)
            x_t = torch.tensor(x_norm_np, dtype=DTYPE_PT).view(1, NX_PHYS, 1)
            u_t = torch.tensor(u_norm_np, dtype=DTYPE_PT).view(1, nu, 1)
            x_norm_np = phy_block(torch.cat([x_t, u_t], dim=1)).view(NX_PHYS).cpu().numpy()

    y_hat = np.array(y_hat_list, dtype=DTYPE_NP)
    return np.sqrt(((y_hat - val_data.y) ** 2).mean(axis=0)) / ystd


def print_nrms(nrms, baseline):
    labels = ['X1', 'X2', 'Y ']
    for ch, lbl in enumerate(labels):
        rms_um = nrms[ch] * ystd[ch] * 1e6
        base_um = baseline[ch] * ystd[ch] * 1e6
        improv = 100.0 * (baseline[ch] - nrms[ch]) / (baseline[ch] + 1e-12)
        print(f'  {lbl}: {nrms[ch]:.4f} ({rms_um:.1f} um)  '
              f'baseline_FP={baseline[ch]:.4f} ({base_um:.1f} um)  '
              f'improvement={improv:+.1f}%')


## ============================================================
## Main
## ============================================================

print('\n' + '='*60)
print('diag21 -- gradient check (na=17, N_CHECK_EPOCHS={})'.format(N_CHECK_EPOCHS))
print('='*60)

print('\nBuilding model...')
fit_sys = build_model(DEFAULT_HP)
na, *_ = _get_encoder_dims(DEFAULT_HP)
print(f'Model built.  na=nb={na}  ({na/FS_NEW*1000:.2f} ms window)')

print('\nComputing baseline FP NRMS...')
baseline_nrms = compute_baseline_fp_nrms()
rms_base = baseline_nrms * ystd * 1e6
print(f'  X1: {baseline_nrms[0]:.4f} ({rms_base[0]:.1f} um)')
print(f'  X2: {baseline_nrms[1]:.4f} ({rms_base[1]:.1f} um)')
print(f'  Y : {baseline_nrms[2]:.4f} ({rms_base[2]:.1f} um)')

print('\n--- Init sim-NRMS (0 gradient steps) ---')
nrms_prev = sim_nrms(fit_sys)
print_nrms(nrms_prev, baseline_nrms)

for epoch in range(1, N_CHECK_EPOCHS + 1):
    print(f'\n--- Training epoch {epoch}/{N_CHECK_EPOCHS} ---')
    fit_sys.fit(
        train_sys_data=train_data,
        val_sys_data=val_data,
        batch_size=DEFAULT_HP['batch_size'],
        epochs=1,
        auto_fit_norm=False,
        loss_kwargs={'nf': DEFAULT_HP['nf']},
        optimizer_kwargs={'lr': DEFAULT_HP['lr']},
        validation_measure="sim-RMS",
    )
    nrms_now = sim_nrms(fit_sys)
    delta = nrms_now - nrms_prev
    print(f'sim-NRMS after epoch {epoch}:')
    print_nrms(nrms_now, baseline_nrms)
    direction = 'IMPROVED' if delta.mean() < 0 else 'DEGRADED'
    print(f'  -> vs previous: dX1={delta[0]:+.4f}  dX2={delta[1]:+.4f}  dY={delta[2]:+.4f}  [{direction}]')
    nrms_prev = nrms_now

print('\n' + '='*60)
print('diag21 complete')
print('='*60)
