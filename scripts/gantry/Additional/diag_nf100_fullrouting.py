"""
diag_nf100_fullrouting.py
-------------------------
Test B: full-state routing (Jan's method) + nf=100 + windowed 100-step validation.
1 trajectory, 1 epoch — enough to see K=0 blowup if it occurs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import deepSI
from scipy.io import loadmat

from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect
from model_augmentation.utils.utils import expansion_matrix, selection_matrix
from model_augmentation.fit_systems.blocks import Static_ANN_Block, Linear_Output_Block, Gantry_State_Block
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.systems.gantry_ss import Cd, Dd, P

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gantry_dynamic.config import RunConfig
from gantry_dynamic.model import PHY_IX
from gantry_dynamic.data import traj_dir

_cfg = RunConfig()
NX_PHYS, nu, ny = _cfg.nx_phys, _cfg.nu, _cfg.ny
TS_NEW   = _cfg.ts_new
D        = _cfg.d
DTYPE_NP = _cfg.dtype_np
DTYPE_PT = _cfg.dtype_pt
TRAJ_DIR = traj_dir(_cfg)

# ── Load 1 training trajectory + 1 val trajectory ──
def load_traj(fname):
    d = loadmat(os.path.join(TRAJ_DIR, fname), squeeze_me=True)
    u = d['u_total'][::D].astype(DTYPE_NP) if 'u_total' in d else d['u'][::D].astype(DTYPE_NP)
    return deepSI.System_data(u=u, y=d['y'][::D].astype(DTYPE_NP), dt=TS_NEW)

train_data = load_traj('T2_X_sym_Y030.mat')
val_data   = load_traj('V1_X_sym_Y_mid_sweep.mat')
print(f'Train: {train_data.u.shape[0]} samples  Val: {val_data.u.shape[0]} samples')

# ── Minimal normalization from 1 trajectory ──
P_inv_T  = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)
pos_log  = (P_inv_T @ train_data.y.T).T
vel_log  = np.vstack([np.diff(pos_log, axis=0) / TS_NEW, np.zeros((1, 3))])
x_all    = np.hstack([pos_log, vel_log])

std_x  = (x_all.std(axis=0).reshape(NX_PHYS, 1) + 1e-8).astype(DTYPE_NP)
std_u  = (train_data.u.std(axis=0).reshape(nu, 1) + 1e-8).astype(DTYPE_NP)
x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
u_mean = train_data.u.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
ystd   = (train_data.y.std(axis=0) + 1e-8).astype(DTYPE_NP)
y0     = train_data.y.mean(axis=0).astype(DTYPE_NP)
Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]

# ── Build model: FULL-STATE routing (no STIFF_IX) ──
NX_ANN = 2
nxd    = NX_PHYS + NX_ANN   # 8
na     = nxd * 2 + 1

ic = Interconnect(nxd, nu, ny, debugging=False)

phy_block = Gantry_State_Block(
    Y_op=None, std_x=std_x, std_u=std_u,
    x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW, up_sample=2,
).to(DTYPE_PT)
out_phys = Linear_Output_Block(C=Cd_norm, D=Dd.numpy())
ann_block = Static_ANN_Block(
    nz=nxd + nu, nw=nxd,
    n_nodes_per_layer=16, n_hidden_layers=2,
    net=zero_init_feed_forward_nn, activation=torch.nn.Tanh,
)

ic.add_block(phy_block)
ic.add_block(out_phys)
ic.add_block(ann_block)

# Full-state routing — Jan's method
ic.connect_block_signals(ann_block, ["x", "u"], ["xp"])
ic.connect_signals("x",  phy_block, "concat",   selection_matrix(PHY_IX, nxd))
ic.connect_block_signals(phy_block, ["u"], [])
ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
ic.connect_signals("x",  out_phys, "concat",    selection_matrix(PHY_IX, nxd))
ic.connect_block_signals(out_phys, ["u"], ["y"])

fit_sys = SSE_Interconnect(
    interconnect=ic, na=na, nb=na, na_right=1, nb_right=1,
    e_net_kwargs={"n_nodes_per_layer": 16, "n_hidden_layers": 2},
)
fit_sys.norm.u0   = u_mean.flatten()
fit_sys.norm.ustd = std_u.flatten()
fit_sys.norm.y0   = y0
fit_sys.norm.ystd = ystd

# ── Train: 1 epoch, nf=100, windowed validation ──
print('\n=== Test B: full-state routing, nf=100, val=100-step-average-RMS, 1 epoch ===\n')
fit_sys.fit(
    train_data, val_data,
    epochs=1,
    loss_kwargs={'nf': 100},
    batch_size=256,
    validation_measure='100-step-average-RMS',
    optimizer_kwargs={'lr': 1e-4},
    auto_fit_norm=False,
    verbose=2,
)

init_val = fit_sys.Loss_val[0]
end_val  = fit_sys.Loss_val[-1]
ratio    = end_val / init_val
print(f'\nInitial val: {init_val:.6f}')
print(f'After epoch: {end_val:.6f}  (x{ratio:.1f})')
print('BLOWUP' if ratio > 5 else 'STABLE')
