"""
diag15_output_aug_diagnostic.py
---------------------------------
Verify output augmentation (y = Cd@x_phys + C_aug@x_aug) as the fix for
the dual constraints that block training of the gantry augmented model.

Root cause summary (diag11, diag13, diag14):
  Constraint 1 (stability):
    2 gantry DT poles exactly at |z|=1 (K[q1]=K[q3]=0, diag14).
    ANN -> x_phys amplifies 400x over nf=400 BPTT rollout -> blowup.
    Gradient clipping does not help (grad_norm already below max_norm, diag13 T_clip).

  Constraint 2 (gradient):
    ANN -> x_aug only, but y = Cd@x_phys ignores x_aug.
    (A_aug, C_aug=0) is unobservable -> ANN grad identically zero (diag11 T1).

Fix:
  Change output equation: y = Cd@x_phys + C_aug@x_aug
  Gradient path: loss -> y -> C_aug -> x_aug -> ANN
  This path never passes through A_phys integrators -> Constraint 1 bypassed.
  C_aug must be nonzero at init for the gradient to be nonzero (Constraint 2 fixed).

Implementation:
  Current:
    out_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)           # C shape (3,6)
    ic.connect_signals("x", out_block, "concat", selection_matrix(PHY_IX, nxd))
  Output aug:
    C_full    = np.hstack([Cd_norm, C_aug_init])                   # C shape (3,8)
    out_block = Linear_Output_Block(C=C_full, D=Dd_np)
    ic.connect_signals("x", out_block, "concat", selection_matrix(np.arange(nxd), nxd))

Note: Linear_Output_Block uses register_buffer (fixed). C_aug is not trainable here.
The gradient path through C_aug to the ANN is sufficient; C_aug training would
require Parameterized_Linear_Output_Block (separate change for gantry_interconnect_dynamic.py).

Tests (all fast, no full training):
  T0  Current architecture, nf=2, 1 backward pass
      Expected: ANN grad ~ 0  (Constraint 2 confirmed)

  T1  Output aug, C_aug physics-init (Y-row nonzero), nf=2, 1 backward pass
      Expected: ANN grad > 0  (Constraint 2 fixed)

  T2  Output aug, C_aug physics-init, nf=400, 1 gradient step
      Expected: val ratio < 1.05  (Constraint 1 maintained: no blowup)

  T3  Output aug, C_aug physics-init, nf=400, 5 gradient steps
      Expected: val stays near baseline (short-term stability)

  T4  Output aug, C_aug = zeros, nf=2, 1 backward pass
      Expected: ANN grad ~ 0  (bootstrapping problem: C_aug=0 kills gradient)
"""

import os
import sys
import time
import numpy as np
import torch
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

N_TRAIN_TRAJ = 1     # fast: 1 trajectory
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

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet',
                       'encoder-augmentation', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

## ═══════════════════════════════════════════════════════════════════════════
## Data
## ═══════════════════════════════════════════════════════════════════════════

TRAJ_DIR    = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab', 'multisine')
TRAIN_FILES = ['T1_Y_sweep_conservative.mat']
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

u_all   = np.concatenate([t.u for t in train_list])
y_all   = np.concatenate([t.y for t in train_list])
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
Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]   # (3, 6)
Dd_np   = Dd.numpy()
PHY_IX  = np.arange(NX_PHYS)

## ═══════════════════════════════════════════════════════════════════════════
## C_aug physics-informed initialization
## ═══════════════════════════════════════════════════════════════════════════
#
# The absorber mass primarily displaces along the Y axis.
# In the normalized model: x_aug[0] ~ delta_a (normalized encoder channel),
# y_norm[2] is the normalized Y position output.
# A small coupling C_aug[2,0] = 1e-2 means 1 unit of x_aug[0] contributes
# 1% of a normalized Y unit. This is physical (absorber displacement is small
# relative to gantry range) and provides a nonzero gradient path.
#
# HEURISTIC: 1e-2 scale chosen so output aug correction does not dominate
# the baseline y = Cd@x_phys at initialization (where ANN output = 0,
# x_aug from encoder is O(1) normalized). Effect is sub-percent.
def make_C_aug_physics(NX_ANN):
    """Physics-informed C_aug init: Y channel receives delta_a (x_aug[0]) weakly."""
    C = np.zeros((ny, NX_ANN), dtype=DTYPE_NP)
    C[2, 0] = 1e-2    # Y <- delta_a (primary absorber coupling to Y axis)
    return C

def make_C_aug_zero(NX_ANN):
    return np.zeros((ny, NX_ANN), dtype=DTYPE_NP)

## ═══════════════════════════════════════════════════════════════════════════
## build_model_common  — shared encoder/interconnect setup
## ═══════════════════════════════════════════════════════════════════════════

def _build_encoder_and_norms(hp, fit_sys):
    """Inject linear_map encoder into fit_sys and set normalisation constants."""
    NX_ANN = hp['NX_ANN']
    na     = 4 * NX_PHYS + 1
    nb     = na

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
    sd = deepSI.System_data(u=u_all, y=y_all)
    sd.x = x_phys_all
    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        Ad, Bd, Cd_dt, Dd_dt, sd)
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
    return fit_sys


## ═══════════════════════════════════════════════════════════════════════════
## build_current  — mirrors gantry_interconnect_dynamic.py (AUG_IX only, no out_aug)
## ═══════════════════════════════════════════════════════════════════════════

def build_current(hp):
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

    # Current: out_block reads only x_phys (PHY_IX=[0..5]), C=(3,6)
    out_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)

    AUG_IX = np.arange(NX_PHYS, nxd)
    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=NX_ANN,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=torch.nn.Tanh,
    )

    ic.add_block(phy_block)
    ic.add_block(out_block)
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
    _build_encoder_and_norms(hp, fit_sys)
    fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
    fit_sys.hfn.to(DTYPE_PT)
    return fit_sys


## ═══════════════════════════════════════════════════════════════════════════
## build_output_aug  — output augmentation: y = Cd@x_phys + C_aug@x_aug
## ═══════════════════════════════════════════════════════════════════════════

def build_output_aug(hp, C_aug_init):
    """
    C_aug_init : (ny, NX_ANN) numpy array — output coupling for x_aug.
    Change vs build_current:
      - C_full = np.hstack([Cd_norm, C_aug_init])    shape (3, 8)
      - out_block gets full x (selection_matrix selects all nxd states)
    """
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

    # Output aug: C_full = [Cd_norm | C_aug_init], shape (3, 8)
    # out_block reads full x[0:8] via selection_matrix(np.arange(nxd), nxd)
    C_full    = np.hstack([Cd_norm, C_aug_init.astype(DTYPE_NP)])  # (3, 8)
    out_block = Linear_Output_Block(C=C_full, D=Dd_np)

    AUG_IX = np.arange(NX_PHYS, nxd)
    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=NX_ANN,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=torch.nn.Tanh,
    )

    ic.add_block(phy_block)
    ic.add_block(out_block)
    ic.add_block(ann_block)

    ic.connect_block_signals(ann_block, ["x", "u"], [])
    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(AUG_IX, nxd))
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    # Key change: select all nxd states so out_block sees x_phys AND x_aug
    ic.connect_signals("x", out_block, "concat", selection_matrix(np.arange(nxd), nxd))
    ic.connect_block_signals(out_block, ["u"], ["y"])

    fit_sys = SSE_Interconnect(
        interconnect=ic, na=na, nb=nb,
        na_right=na_right, nb_right=nb_right,
        e_net_kwargs={
            "n_nodes_per_layer": hp['n_nodes_per_layer'],
            "n_hidden_layers": hp['n_hidden_layers'],
        },
    )
    _build_encoder_and_norms(hp, fit_sys)
    fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
    fit_sys.hfn.to(DTYPE_PT)
    return fit_sys


## ═══════════════════════════════════════════════════════════════════════════
## Helpers
## ═══════════════════════════════════════════════════════════════════════════

def fresh(builder, *args):
    """Build a fresh model with fixed seeds."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    return builder(DEFAULT_HP, *args)

def get_val_rms(m):
    m.eval()
    return float(m.cal_validation_error(val_data, validation_measure='sim-RMS'))

def get_ann_grad(m):
    """Return final-layer weight gradient norm for the ANN block."""
    ann = next(b for b in m.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    final = ann.net.net[-1]
    g = final.weight.grad
    return g.norm().item() if g is not None else 0.0

def backward_pass(m, nf):
    """Single forward+backward pass on training data. Returns loss."""
    data  = m.make_training_data(m.norm.transform(train_data), nf=nf)
    n     = len(data[0])
    rng   = np.random.default_rng(SEED)
    idx   = rng.choice(n, DEFAULT_HP['batch_size'], replace=False)
    batch = [torch.tensor(data[i][idx], dtype=DTYPE_PT) for i in range(len(data))]
    m.train()
    m.optimizer.zero_grad()
    loss = m.loss(*batch, nf=nf)
    loss.backward()
    return loss

def gradient_step(m, nf):
    """Single optimizer step. Returns loss value."""
    loss = backward_pass(m, nf)
    m.optimizer.step()
    return loss

def enc_grad_norm(m):
    return sum(
        p.grad.detach().norm().item()**2
        for p in m.encoder.parameters() if p.grad is not None
    ) ** 0.5


## ═══════════════════════════════════════════════════════════════════════════
## T0 — Current architecture: confirm ANN grad = 0
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('T0: Current architecture — gradient path (nf=2)')
print('Expected: ANN grad ~ 0 (Constraint 2: y=Cd@x_phys ignores x_aug)')
print('='*60)
t0 = time.time()

m0 = fresh(build_current)
backward_pass(m0, nf=2)

g_ann0 = get_ann_grad(m0)
g_enc0 = enc_grad_norm(m0)
print(f'  ANN final-layer grad : {g_ann0:.3e}')
print(f'  Encoder grad         : {g_enc0:.3e}')
if g_ann0 < 1e-10:
    print('  [RESULT] ANN DISCONNECTED (grad~0). Constraint 2 confirmed.')
else:
    print(f'  [RESULT] Unexpected: ANN grad = {g_ann0:.3e} (should be 0).')
print(f'  ({time.time()-t0:.1f}s)')


## ═══════════════════════════════════════════════════════════════════════════
## T1 — Output aug (C_aug physics-init): verify ANN gradient path (nf=2)
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('T1: Output aug (C_aug physics-init) — gradient path (nf=2)')
print('Expected: ANN grad > 0 (Constraint 2 fixed: y reads x_aug via C_aug)')
print('='*60)
t0 = time.time()

C_aug_phy = make_C_aug_physics(DEFAULT_HP['NX_ANN'])
print(f'  C_aug_init:\n{C_aug_phy}')

m1 = fresh(build_output_aug, C_aug_phy)
backward_pass(m1, nf=2)

g_ann1 = get_ann_grad(m1)
g_enc1 = enc_grad_norm(m1)
print(f'  ANN final-layer grad : {g_ann1:.3e}')
print(f'  Encoder grad         : {g_enc1:.3e}')
if g_ann1 > 1e-10:
    print(f'  [RESULT] ANN CONNECTED (grad={g_ann1:.3e}). Constraint 2 fixed.')
else:
    print('  [RESULT] ANN still disconnected. Check C_aug init and encoder x_aug.')
print(f'  ({time.time()-t0:.1f}s)')


## ═══════════════════════════════════════════════════════════════════════════
## T2 — Output aug (C_aug physics-init): blowup check at nf=400 (1 step)
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('T2: Output aug (C_aug physics-init) — blowup check (nf=400, 1 step)')
print('Expected: val ratio < 1.05 (Constraint 1 maintained: grad via C_aug, not A_phys)')
print('='*60)
t0 = time.time()

m2 = fresh(build_output_aug, C_aug_phy)
val_before2 = get_val_rms(m2)

backward_pass(m2, nf=400)
gnorm_hfn2 = sum(
    p.grad.detach().norm().item()**2
    for p in m2.hfn.parameters() if p.grad is not None
) ** 0.5
m2.optimizer.step()

val_after2 = get_val_rms(m2)
ratio2 = val_after2 / val_before2
flag2  = '^ WORSE' if ratio2 > 1.05 else ('v better' if ratio2 < 0.95 else '~ stable')

print(f'  val: {val_before2:.5f} -> {val_after2:.5f}  {flag2}  (x{ratio2:.2f})')
print(f'  hfn_grad: {gnorm_hfn2:.3e}')
if ratio2 < 1.05:
    print('  [RESULT] No blowup. Output aug maintains stability at nf=400.')
else:
    print(f'  [RESULT] Blowup (x{ratio2:.1f}). Gradient still reaches A_phys integrators.')
print(f'  ({time.time()-t0:.1f}s)')


## ═══════════════════════════════════════════════════════════════════════════
## T3 — Output aug (C_aug physics-init): 5-step stability at nf=400
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('T3: Output aug (C_aug physics-init) — 5-step stability (nf=400)')
print('Expected: val stays near initial value (no drift over short training)')
print('='*60)
t0 = time.time()

m3 = fresh(build_output_aug, C_aug_phy)
val_init3 = get_val_rms(m3)
print(f'  Step 0 (init): val = {val_init3:.5f}')

ratios = []
for step in range(1, 6):
    gradient_step(m3, nf=400)
    v = get_val_rms(m3)
    r = v / val_init3
    ratios.append(r)
    flag = '^ WORSE' if r > 1.05 else ('v better' if r < 0.95 else '~ stable')
    print(f'  Step {step}: val = {v:.5f}  x{r:.2f}  {flag}')

max_ratio = max(ratios)
if max_ratio < 1.05:
    print(f'  [RESULT] Stable over 5 steps (max ratio = {max_ratio:.2f}). Short-term training safe.')
elif max_ratio < 2.0:
    print(f'  [RESULT] Moderate degradation (max ratio = {max_ratio:.2f}). Monitor carefully.')
else:
    print(f'  [RESULT] Blowup over 5 steps (max ratio = {max_ratio:.2f}). Fix insufficient.')
print(f'  ({time.time()-t0:.1f}s)')


## ═══════════════════════════════════════════════════════════════════════════
## T4 — Output aug (C_aug = zeros): confirm bootstrapping problem
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('T4: Output aug (C_aug = zeros) — bootstrapping problem (nf=2)')
print('Expected: ANN grad ~ 0  (zero C_aug kills gradient path to ANN)')
print('='*60)
t0 = time.time()

C_aug_zero = make_C_aug_zero(DEFAULT_HP['NX_ANN'])
m4 = fresh(build_output_aug, C_aug_zero)
backward_pass(m4, nf=2)

g_ann4 = get_ann_grad(m4)
g_enc4 = enc_grad_norm(m4)
print(f'  ANN final-layer grad : {g_ann4:.3e}')
print(f'  Encoder grad         : {g_enc4:.3e}')
if g_ann4 < 1e-10:
    print('  [RESULT] ANN DISCONNECTED (grad~0). Bootstrapping problem confirmed.')
    print('           C_aug MUST be initialized nonzero for gradient to reach ANN.')
else:
    print(f'  [RESULT] ANN connected even with C_aug=0 (grad={g_ann4:.3e}).')
    print('           Check: some other gradient path may exist.')
print(f'  ({time.time()-t0:.1f}s)')


## ═══════════════════════════════════════════════════════════════════════════
## Summary
## ═══════════════════════════════════════════════════════════════════════════

print()
print('='*60)
print('SUMMARY')
print('='*60)
print(f'  T0 (current baseline)  : ANN grad = {g_ann0:.3e}  (expected ~ 0)')
print(f'  T1 (output aug, C≠0)   : ANN grad = {g_ann1:.3e}  (expected > 0)')
print(f'  T2 (output aug, nf=400): val ratio = {ratio2:.3f}  (expected < 1.05)')
print(f'  T3 (5 steps, nf=400)   : max ratio = {max_ratio:.3f}  (expected < 1.05)')
print(f'  T4 (output aug, C=0)   : ANN grad = {g_ann4:.3e}  (expected ~ 0)')

# Decision criteria
t1_ok = g_ann1 > 1e-10
t2_ok = ratio2 < 1.05
t3_ok = max_ratio < 1.05
t4_ok = g_ann4 < 1e-10   # bootstrapping confirmed as expected

print()
if t1_ok and t2_ok and t3_ok and t4_ok:
    print('  ALL TESTS PASS.')
    print('  Output augmentation with nonzero C_aug_init is a valid fix.')
    print()
    print('  Next steps:')
    print('    1. Add C_aug as trainable nn.Parameter to Linear_Output_Block')
    print('       or use Parameterized_Linear_Output_Block variant')
    print('    2. Update gantry_interconnect_dynamic.py with output aug routing')
    print('    3. Run full training (10+ epochs) and compare val NRMS to baseline FP')
else:
    print('  SOME TESTS FAILED:')
    if not t1_ok:
        print('    - T1 FAIL: ANN still disconnected with nonzero C_aug.')
        print('      Check encoder x_aug channels and BPTT graph.')
    if not t2_ok:
        print(f'    - T2 FAIL: Blowup at nf=400 (ratio={ratio2:.1f}).')
        print('      Gradient may still reach A_phys through another path.')
    if not t3_ok:
        print(f'    - T3 FAIL: Blowup over 5 steps (max ratio={max_ratio:.1f}).')
    if not t4_ok:
        print(f'    - T4 UNEXPECTED: ANN still has gradient with C_aug=0.')
        print('      There is another gradient path not accounted for.')

## ═══════════════════════════════════════════════════════════════════════════
## Plot: val trajectory over T3 steps
## ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 3.5))
steps_plot = [0] + list(range(1, 6))
vals_plot  = [val_init3] + [val_init3 * r for r in ratios]
ax.plot(steps_plot, vals_plot, 'o-', color='C0', label='Output aug (C_aug physics-init)')
ax.axhline(val_init3, color='k', lw=0.8, linestyle='--', label='Init val')
ax.axhline(val_init3 * 1.05, color='C3', lw=0.8, linestyle=':', label='+5% threshold')
ax.set_xlabel('Gradient step')
ax.set_ylabel('Val sim-RMS')
ax.set_title('T3: 5-step stability check (nf=400, output aug)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plot_path = os.path.join(OUT_DIR, 'diag15_5step_stability.png')
fig.savefig(plot_path, dpi=150)
print(f'\nSaved plot: {plot_path}')
print('Done.')
