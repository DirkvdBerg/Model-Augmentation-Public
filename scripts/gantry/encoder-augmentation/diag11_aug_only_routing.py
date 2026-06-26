"""
diag11_aug_only_routing.py
--------------------------
Fast sanity check for the new ANN routing:
  ANN output -> augmented states [6,7] only (via expansion_matrix)

Two checks:
  T1  Gradient path  (nf=2, 1 backward pass, no weight update)
      -> grad_total of ANN final layer (all rows are now aug rows)
      -> Are ANN weights reachable from the loss at all?

  T2  Blowup check   (nf=400, 1 gradient step)
      -> val before / after
      -> Does restricting to aug rows prevent instability?

Expected outcomes:
  T1  grad_total > 0 -> ANN still connected to loss (good)
      grad_total ~ 0 -> ANN fully disconnected from loss (bad: x_aug has
                        no path to y=Cd@x_phys, need to also route to
                        velocity rows [3,4,5])

  T2  val stays near epoch-0 baseline (~0.002) -> routing fix works

Fast: 1 training trajectory, ~30-60s total.
"""

import os, sys, time
import numpy as np
import torch
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import deepSI
from model_augmentation.utils.utils import normalize_linear_ss_matrices
from model_augmentation.utils.utils import expansion_matrix, selection_matrix
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import SSE_Interconnect, Interconnect
from model_augmentation.fit_systems.blocks import (
    Gantry_State_Block, Linear_Output_Block, Static_ANN_Block,
)
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize

## ═══════════════════════════════════════════════════════════════════════════
## Config
## ═══════════════════════════════════════════════════════════════════════════

N_TRAIN_TRAJ = 1     # fast: 1 trajectory only
SEED         = 42
NX_PHYS      = 6
nu           = 3
ny           = 3
Y_OP         = None

FS_ORIG = 20000
FS_NEW  = 4000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32

DEFAULT_HP = dict(
    NX_ANN=2,
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=2,
    nf=400,
    na_nb=0,
    batch_size=256,
    lr=1e-4,
    epochs=10,
)
DEFAULT_HP['na_nb'] = (NX_PHYS + DEFAULT_HP['NX_ANN']) * 2 + 1

## ═══════════════════════════════════════════════════════════════════════════
## Data
## ═══════════════════════════════════════════════════════════════════════════

TRAJ_DIR = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab', 'multisine')
TRAIN_FILES = ['T1_Y_sweep_conservative.mat']   # 1 trajectory
VAL_FILE    = 'V1_X_sym_Y_mid_sweep.mat'

def _load_u(d):
    return d['u_total'] if 'u_total' in d else d['u']

def load_traj(f):
    d = loadmat(os.path.join(TRAJ_DIR, f), squeeze_me=True)
    return deepSI.System_data(
        u=_load_u(d)[::D].astype(DTYPE_NP),
        y=d['y'][::D].astype(DTYPE_NP),
        dt=TS_NEW,
    )

print(f'Loading data ({N_TRAIN_TRAJ} train traj)...')
train_list = [load_traj(f) for f in TRAIN_FILES[:N_TRAIN_TRAJ]]
train_data = deepSI.System_data_list(train_list)
val_data   = load_traj(VAL_FILE)
print(f'  train: {sum(len(t.y) for t in train_list)} samples | val: {len(val_data.y)} samples')

## ═══════════════════════════════════════════════════════════════════════════
## Normalisation
## ═══════════════════════════════════════════════════════════════════════════

u_all = np.concatenate([t.u for t in train_list])
y_all = np.concatenate([t.y for t in train_list])
P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)

x_logical_list = []
for t in train_list:
    pos = (P_inv_T @ t.y.T).T
    vel = np.diff(pos, axis=0) * (1.0 / train_list[0].dt)
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

## ═══════════════════════════════════════════════════════════════════════════
## build_model  — mirrors the updated gantry_interconnect_dynamic.py
## ═══════════════════════════════════════════════════════════════════════════

def build_model(hp):
    NX_ANN = hp['NX_ANN']
    nxd    = NX_PHYS + NX_ANN
    na     = 4 * NX_PHYS + 1
    nb     = na
    na_right = nb_right = 1

    ic = Interconnect(nxd, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
        up_sample=hp['up_sample'],
    ).to(DTYPE_PT)
    out_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    ic.add_block(phy_block)
    ic.add_block(out_block)

    AUG_IX    = np.arange(NX_PHYS, nxd)
    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=NX_ANN,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=torch.nn.Tanh,
    )
    ic.add_block(ann_block)

    ic.connect_block_signals(ann_block, ["x", "u"], [])
    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(AUG_IX, nxd))
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", out_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_block, ["u"], ["y"])

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

    Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)
    baseline_npz = os.path.join(
        PROJECT_ROOT, 'data', 'gantry', 'baseline_simulations',
        'multisine_LPV', 'baseline_states.npz')
    x_phys_all = (np.concatenate(np.load(baseline_npz, allow_pickle=True)['x_train_phys'])
                  if os.path.exists(baseline_npz) else x_all)
    sd = deepSI.System_data(u=u_all, y=y_all); sd.x = x_phys_all
    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(Ad, Bd, Cd_dt, Dd_dt, sd)
    fit_sys.encoder = linear_encoder_init_aug(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
        nx_aug=NX_ANN,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        flag_linear_only=False,
        u_mean=u_mean, std_u=std_u,
        y0=y0, ystd=ystd, x_mean=x_mean, std_x=std_x,
    ).to(DTYPE_PT)

    fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
    fit_sys.hfn.to(DTYPE_PT)
    return fit_sys

def build_fresh():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    return build_model(DEFAULT_HP)

def get_val_sim_rms(m):
    m.eval()
    return float(m.cal_validation_error(val_data, validation_measure='sim-RMS'))

## ═══════════════════════════════════════════════════════════════════════════
## T1: Gradient path check  (nf=2, single backward pass)
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('T1: Gradient path (nf=2, 1 backward pass)')
print('='*60)
t0 = time.time()

rng   = np.random.default_rng(SEED)
m1    = build_fresh()
data1 = m1.make_training_data(m1.norm.transform(train_data), nf=2)

n1    = len(data1[0])
idx1  = rng.choice(n1, DEFAULT_HP['batch_size'], replace=False)
batch1 = [torch.tensor(data1[i][idx1], dtype=DTYPE_PT) for i in range(len(data1))]

m1.train()
m1.optimizer.zero_grad()
loss1 = m1.loss(*batch1, nf=2)
loss1.backward()

ann1 = next(b for b in m1.hfn.connected_blocks if isinstance(b, Static_ANN_Block))

print(f'  sqrt_loss = {float(loss1.item())**0.5:.4e}')
print('  ANN layer gradients:')
for name, p in ann1.net.net.named_parameters():
    g = p.grad.norm().item() if p.grad is not None else 0.0
    print(f'    {name:30s}  {g:.3e}  {list(p.shape)}')

final1   = ann1.net.net[-1]
W_grad1  = final1.weight.grad
grad_total = W_grad1.norm().item() if W_grad1 is not None else 0.0
print(f'\n  Final layer W_grad shape: {list(final1.weight.shape)}  (nw=NX_ANN={DEFAULT_HP["NX_ANN"]})')
print(f'  grad_total: {grad_total:.3e}')

gnorm_enc1 = sum(
    p.grad.detach().norm().item()**2
    for p in m1.encoder.parameters() if p.grad is not None
)**0.5
print(f'  enc_grad:   {gnorm_enc1:.3e}')

if grad_total < 1e-10:
    print('\n  [RESULT] grad_total ~ 0 -> ANN DISCONNECTED from loss.')
    print('           x_aug has no path to y=Cd@x_phys.')
    print('           Fix: also route ANN to velocity rows [3,4,5].')
else:
    print(f'\n  [RESULT] grad_total = {grad_total:.3e} -> ANN connected to loss. OK.')
print(f'  ({time.time()-t0:.0f}s)')

## ═══════════════════════════════════════════════════════════════════════════
## T2: Blowup check  (nf=400, 1 gradient step)
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('T2: Blowup check (nf=400, 1 gradient step)')
print('='*60)
t0 = time.time()

rng2  = np.random.default_rng(SEED)
m2    = build_fresh()
val_before = get_val_sim_rms(m2)

data2  = m2.make_training_data(m2.norm.transform(train_data), nf=400)
n2     = len(data2[0])
idx2   = rng2.choice(n2, DEFAULT_HP['batch_size'], replace=False)
batch2 = [torch.tensor(data2[i][idx2], dtype=DTYPE_PT) for i in range(len(data2))]

m2.train()
m2.optimizer.zero_grad()
loss2 = m2.loss(*batch2, nf=400)
loss2.backward()

gnorm_hfn2 = sum(
    p.grad.detach().norm().item()**2
    for p in m2.hfn.parameters() if p.grad is not None
)**0.5
m2.optimizer.step()
val_after = get_val_sim_rms(m2)

ratio = val_after / val_before
flag  = '^ WORSE' if ratio > 1.05 else ('v better' if ratio < 0.95 else '~ stable')
print(f'  val: {val_before:.5f} -> {val_after:.5f}  {flag}  (x{ratio:.2f})')
print(f'  hfn_grad: {gnorm_hfn2:.3e}')

if ratio < 1.05:
    print('\n  [RESULT] No blowup. Routing fix prevents physical state corruption.')
else:
    print(f'\n  [RESULT] Still blowing up (x{ratio:.1f}). Routing fix insufficient.')
print(f'  ({time.time()-t0:.0f}s)')
