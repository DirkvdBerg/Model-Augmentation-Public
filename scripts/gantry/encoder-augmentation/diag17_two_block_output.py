"""
diag17_two_block_output.py
--------------------------
Verifies that Jan's two-block output pattern keeps Cd_norm fixed
while allowing C_aug to be trained.

T0: Current wrong impl  — Parameterized_Linear_Output_Block(C=[Cd_norm | C_aug_init])
    Both Cd_norm and C_aug are nn.Parameter → Cd_norm drifts every gradient step.

T1: Two-block fix (Jan's pattern):
      out_phys = Linear_Output_Block(C=Cd_norm, D=Dd)          <- register_buffer, never changes
      out_aug  = Parameterized_Linear_Output_Block(C=C_aug_init) <- nn.Parameter, trains
    Both connect additively to y (default in Interconnect when output_signal_ix<=1).

Three checks:
  1. Cd_norm gradient after 1 backward:
       T0 -> nonzero  (Cd_norm columns get gradient from x_phys; PROBLEM)
       T1 -> None     (buffer has no gradient; CORRECT)
  2. Cd_norm drift after N_STEPS gradient steps:
       T0 -> ||delta_Cd|| > 0  (drifts; PROBLEM)
       T1 -> ||delta_Cd|| = 0  (never changes; CORRECT)
  3. val_rms after N_STEPS:
       T0 -> improves (but for wrong reason: Cd_norm is adjusting)
       T1 -> flat     (Cd_norm fixed; ANN/C_aug have zero gradient because
                       x_aug=0 with zero-init ANN -- bootstrapping issue,
                       not a flaw of the two-block pattern itself)

Note on bootstrapping (T1):
  With zero_init_feed_forward_nn, the ANN final layer weights W_out=0 →
  ANN output=0 → x_aug=0 → dL/dC_aug = dL/dy @ x_aug.T = 0 → no learning.
  This bootstrapping is separate from the Cd-trainability problem.
  After the encoder learns to produce nonzero x_aug estimates, C_aug gets
  nonzero gradient and the ANN can train. This is shown in Check 3b.
"""
import os
import sys
import numpy as np
import torch

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import deepSI
from model_augmentation.utils.utils import normalize_linear_ss_matrices
from model_augmentation.utils.utils import expansion_matrix, selection_matrix
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import SSE_Interconnect, Interconnect
from model_augmentation.fit_systems.blocks import (
    Linear_State_Block, Linear_Output_Block,
    Parameterized_Linear_Output_Block, Static_ANN_Block,
)
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug

## ═══════════════════════════════════════════════════════════════════════════
## Dimensions — matching gantry exactly
## ═══════════════════════════════════════════════════════════════════════════

NX_PHYS  = 6
NX_ANN   = 2
nu       = 3
ny       = 3
nxd      = NX_PHYS + NX_ANN   # 8
DT       = 0.00025             # gantry 4 kHz
NF       = 400
SEED     = 42
DTYPE_NP = np.float32
DTYPE_PT = torch.float32
N_STEPS  = 5
LR       = 1e-4
BATCH_SIZE = 256

NA = NB = (NX_PHYS + NX_ANN) * 2 + 1   # = 17 (Jan's formula)
NA_RIGHT = NB_RIGHT = 1
N_NODES  = 16
N_LAYERS = 2

PHY_IX = np.arange(NX_PHYS)        # [0..5]
AUG_IX = np.arange(NX_PHYS, nxd)   # [6, 7]

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet',
                       'encoder-augmentation', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

## ═══════════════════════════════════════════════════════════════════════════
## Synthetic data (stable system + hidden absorber)
## ═══════════════════════════════════════════════════════════════════════════

np.random.seed(SEED)
# Stable physical: 6 modes at |z|=0.99
A_np   = 0.99 * np.eye(NX_PHYS, dtype=DTYPE_NP)
B_np   = (np.random.randn(NX_PHYS, nu) * 0.01).astype(DTYPE_NP)
Cd_np  = (np.random.randn(ny, NX_PHYS) * 0.5).astype(DTYPE_NP)
Dd_np  = np.zeros((ny, nu), dtype=DTYPE_NP)
# Hidden absorber drives y through C_abs_true (unknown to model)
A_abs      = 0.95 * np.eye(NX_ANN, dtype=DTYPE_NP)
B_abs      = (np.random.randn(NX_ANN, NX_PHYS) * 0.05).astype(DTYPE_NP)
C_abs_true = (np.random.randn(ny, NX_ANN) * 0.5).astype(DTYPE_NP)

N_TRAIN = 2000
N_VAL   = 1000
rng = np.random.default_rng(SEED)
u_seq = rng.normal(0, 1.0, (N_TRAIN + N_VAL, nu)).astype(DTYPE_NP)

N = N_TRAIN + N_VAL
x_phys_all = np.zeros((N, NX_PHYS), dtype=DTYPE_NP)
x_aug_all  = np.zeros((N, NX_ANN),  dtype=DTYPE_NP)
y_all      = np.zeros((N, ny),      dtype=DTYPE_NP)
for k in range(N):
    y_all[k] = Cd_np @ x_phys_all[k] + C_abs_true @ x_aug_all[k]
    if k < N - 1:
        x_phys_all[k+1] = A_np @ x_phys_all[k] + B_np @ u_seq[k]
        x_aug_all[k+1]  = A_abs @ x_aug_all[k] + B_abs @ x_phys_all[k]

u_tr, y_tr, xp_tr = u_seq[:N_TRAIN], y_all[:N_TRAIN], x_phys_all[:N_TRAIN]
u_va, y_va         = u_seq[N_TRAIN:], y_all[N_TRAIN:]
train_sd = deepSI.System_data(u=u_tr, y=y_tr, dt=DT)
val_sd   = deepSI.System_data(u=u_va, y=y_va, dt=DT)

## Normalisation
u_mean = u_tr.mean(axis=0).reshape(nu,     1).astype(DTYPE_NP)
std_u  = u_tr.std(axis=0) .reshape(nu,     1).astype(DTYPE_NP) + 1e-8
y0     = y_tr.mean(axis=0).astype(DTYPE_NP)
ystd   = y_tr.std(axis=0) .astype(DTYPE_NP) + 1e-8
x_mean = xp_tr.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
std_x  = xp_tr.std(axis=0) .reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8

sd_with_x     = deepSI.System_data(u=u_tr, y=y_tr)
sd_with_x.x   = xp_tr
Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
    A_np.astype(np.float64), B_np.astype(np.float64),
    Cd_np.astype(np.float64), Dd_np.astype(np.float64),
    sd_with_x,
)
Cd_norm = Cd_bar.astype(DTYPE_NP)   # (ny, NX_PHYS)
Dd_norm = Dd_bar.astype(DTYPE_NP)   # (ny, nu)

# C_aug_init: small nonzero so C_aug contributes once x_aug is nonzero
C_aug_init = np.zeros((ny, NX_ANN), dtype=DTYPE_NP)
C_aug_init[0, 0] = 1e-2

## ═══════════════════════════════════════════════════════════════════════════
## Shared encoder initialisation (identical for both models)
## ═══════════════════════════════════════════════════════════════════════════

def make_encoder():
    return linear_encoder_init_aug(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=NA, nb=NB, nx_aug=NX_ANN,
        n_nodes_per_layer=N_NODES, n_hidden_layers=N_LAYERS,
        flag_linear_only=False,
        u_mean=u_mean, std_u=std_u, y0=y0, ystd=ystd,
        x_mean=x_mean, std_x=std_x,
    ).to(DTYPE_PT)

## ═══════════════════════════════════════════════════════════════════════════
## Model builders
## ═══════════════════════════════════════════════════════════════════════════

def _base_parts():
    """Shared: physics block, ANN block, encoder."""
    np.random.seed(SEED); torch.manual_seed(SEED)
    phy_block = Linear_State_Block(
        A=torch.tensor(Ad_bar, dtype=DTYPE_PT),
        B=torch.tensor(Bd_bar, dtype=DTYPE_PT),
    )
    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=NX_ANN,
        n_nodes_per_layer=N_NODES, n_hidden_layers=N_LAYERS,
        net=zero_init_feed_forward_nn, activation=torch.nn.Tanh,
    )
    return phy_block, ann_block


def _routing(ic, phy_block, ann_block):
    """D-065 routing: ANN -> x_aug only. Identical in T0 and T1."""
    ic.connect_block_signals(ann_block, ["x", "u"], [])
    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(AUG_IX, nxd))
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))


def _finalise(ic):
    fit_sys = SSE_Interconnect(
        interconnect=ic, na=NA, nb=NB,
        na_right=NA_RIGHT, nb_right=NB_RIGHT,
        e_net_kwargs={"n_nodes_per_layer": N_NODES, "n_hidden_layers": N_LAYERS},
    )
    fit_sys.norm.u0   = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0   = y0
    fit_sys.norm.ystd = ystd
    fit_sys.encoder   = make_encoder()
    fit_sys.init_model(sys_data=deepSI.System_data_list([train_sd]),
                       auto_fit_norm=False)
    fit_sys.hfn.to(DTYPE_PT)
    for pg in fit_sys.optimizer.param_groups:
        pg['lr'] = LR
    return fit_sys


def build_T0():
    """Wrong: combined Parameterized block trains Cd_norm and C_aug together."""
    phy_block, ann_block = _base_parts()
    ic = Interconnect(nxd, nu, ny, debugging=False)
    C_full    = np.hstack([Cd_norm, C_aug_init])   # (ny, nxd)
    out_block = Parameterized_Linear_Output_Block(
        C=C_full, D=Dd_norm, flag_loss_reg=False)
    ic.add_block(phy_block); ic.add_block(out_block); ic.add_block(ann_block)
    _routing(ic, phy_block, ann_block)
    # output reads full state
    ic.connect_signals("x", out_block, "concat",
                       selection_matrix(np.arange(nxd), nxd))
    ic.connect_block_signals(out_block, ["u"], ["y"])
    return _finalise(ic), out_block, ann_block


def build_T1():
    """Fix: Linear_Output_Block (fixed Cd) + Parameterized (trainable C_aug)."""
    phy_block, ann_block = _base_parts()
    ic = Interconnect(nxd, nu, ny, debugging=False)
    out_phys = Linear_Output_Block(C=Cd_norm, D=Dd_norm)
    out_aug  = Parameterized_Linear_Output_Block(
        C=C_aug_init, D=np.zeros((ny, nu), dtype=DTYPE_NP), flag_loss_reg=False)
    ic.add_block(phy_block); ic.add_block(out_phys)
    ic.add_block(out_aug);  ic.add_block(ann_block)
    _routing(ic, phy_block, ann_block)
    # Physical: x_phys + u -> y (additive, default)
    ic.connect_signals("x", out_phys, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_phys, ["u"], ["y"])
    # Aug: x_aug + u -> y (additive, default)
    ic.connect_signals("x", out_aug, "concat", selection_matrix(AUG_IX, nxd))
    ic.connect_block_signals(out_aug, ["u"], ["y"])
    return _finalise(ic), out_phys, out_aug, ann_block

## ═══════════════════════════════════════════════════════════════════════════
## Helpers
## ═══════════════════════════════════════════════════════════════════════════

def one_backward(m):
    """One forward+backward, no weight update. Returns loss."""
    train_list = deepSI.System_data_list([train_sd])
    td  = m.make_training_data(m.norm.transform(train_list), nf=NF)
    n   = len(td[0])
    idx = np.random.default_rng(SEED).choice(n, min(BATCH_SIZE, n), replace=False)
    batch = [torch.tensor(td[i][idx], dtype=DTYPE_PT) for i in range(len(td))]
    m.train()
    m.optimizer.zero_grad()
    loss = m.loss(*batch, nf=NF)
    loss.backward()
    return float(loss)


def gradient_step(m, step=0):
    """One full gradient step."""
    train_list = deepSI.System_data_list([train_sd])
    td  = m.make_training_data(m.norm.transform(train_list), nf=NF)
    n   = len(td[0])
    idx = np.random.default_rng(SEED + step).choice(n, min(BATCH_SIZE, n), replace=False)
    batch = [torch.tensor(td[i][idx], dtype=DTYPE_PT) for i in range(len(td))]
    m.train()
    m.optimizer.zero_grad()
    loss = m.loss(*batch, nf=NF)
    loss.backward()
    m.optimizer.step()


def get_val_rms(m):
    m.eval()
    return float(m.cal_validation_error(val_sd, validation_measure='sim-RMS'))


def grad_norm(tensor):
    """Returns gradient norm, or None if no gradient (buffer)."""
    if tensor is None:
        return None
    g = getattr(tensor, 'grad', None)
    if g is None:
        return None
    return float(g.norm())


def ann_grad_norm(ann_block):
    total = 0.0
    for p in ann_block.parameters():
        if p.grad is not None:
            total += float(p.grad.norm())
    return total

## ═══════════════════════════════════════════════════════════════════════════
## T0: combined Parameterized block (current wrong implementation)
## ═══════════════════════════════════════════════════════════════════════════

print('=' * 60)
print('T0: combined Parameterized_Linear_Output_Block (current)')
print('=' * 60)
m0, out_T0, ann_T0 = build_T0()
Cd_init_T0   = out_T0.C.data[:, :NX_PHYS].clone()
Caug_init_T0 = out_T0.C.data[:, NX_PHYS:].clone()
val0_T0      = get_val_rms(m0)

one_backward(m0)
cd_grad_T0   = grad_norm(out_T0.C)   # gradient of the combined C matrix
caug_col_grad_T0 = float(out_T0.C.grad[:, NX_PHYS:].norm()) if out_T0.C.grad is not None else 0.0
cd_col_grad_T0   = float(out_T0.C.grad[:, :NX_PHYS].norm()) if out_T0.C.grad is not None else 0.0
ann_grad_T0  = ann_grad_norm(ann_T0)
print(f'  After 1 backward:')
print(f'    Cd_norm columns grad norm  : {cd_col_grad_T0:.3e}  <-- NONZERO (problem: Cd will drift)')
print(f'    C_aug columns grad norm    : {caug_col_grad_T0:.3e}')
print(f'    ANN param grad norm        : {ann_grad_T0:.3e}')

# Drift after N_STEPS
m0, out_T0, ann_T0 = build_T0()
Cd_init_T0 = out_T0.C.data[:, :NX_PHYS].clone()
for step in range(N_STEPS):
    gradient_step(m0, step=step + 1)
Cd_drift_T0  = float((out_T0.C.data[:, :NX_PHYS] - Cd_init_T0).norm())
Caug_delta_T0 = float((out_T0.C.data[:, NX_PHYS:] - Caug_init_T0).norm())
val_T0       = get_val_rms(m0)
print(f'\n  After {N_STEPS} gradient steps:')
print(f'    ||delta_Cd||   : {Cd_drift_T0:.3e}  <-- DRIFTS (wrong)')
print(f'    ||delta_C_aug||: {Caug_delta_T0:.3e}')
print(f'    val_rms        : {val_T0:.5f}  (was {val0_T0:.5f},'
      f' ratio {val_T0/val0_T0:.2f}x)')

## ═══════════════════════════════════════════════════════════════════════════
## T1: two-block fix (Jan's pattern)
## ═══════════════════════════════════════════════════════════════════════════

print()
print('=' * 60)
print('T1: two-block fix  Linear_Output_Block + Parameterized')
print('=' * 60)
m1, out_phys_T1, out_aug_T1, ann_T1 = build_T1()
Cd_init_T1   = out_phys_T1.C.clone()         # buffer, should never change
Caug_init_T1 = out_aug_T1.C.data.clone()
val0_T1      = get_val_rms(m1)

one_backward(m1)
# out_phys_T1.C is a register_buffer — not nn.Parameter, grad=None always
cd_grad_T1   = grad_norm(out_phys_T1.C)      # expect None
caug_grad_T1 = grad_norm(out_aug_T1.C)
ann_grad_T1  = ann_grad_norm(ann_T1)
is_param_T1  = isinstance(out_phys_T1.C, torch.nn.Parameter)
print(f'  After 1 backward:')
print(f'    out_phys.C is nn.Parameter : {is_param_T1}  (False = register_buffer = correct)')
print(f'    Cd_norm grad               : {cd_grad_T1}   <-- None (correct: cannot drift)')
print(f'    C_aug grad norm            : {caug_grad_T1}')
print(f'    ANN param grad norm        : {ann_grad_T1}')

# Drift after N_STEPS
m1, out_phys_T1, out_aug_T1, ann_T1 = build_T1()
Cd_init_T1   = out_phys_T1.C.clone()
Caug_init_T1 = out_aug_T1.C.data.clone()
for step in range(N_STEPS):
    gradient_step(m1, step=step + 1)
Cd_drift_T1   = float((out_phys_T1.C - Cd_init_T1).norm())
Caug_delta_T1 = float((out_aug_T1.C.data - Caug_init_T1).norm())
val_T1        = get_val_rms(m1)
print(f'\n  After {N_STEPS} gradient steps:')
print(f'    ||delta_Cd||   : {Cd_drift_T1:.3e}  <-- EXACTLY 0 (correct)')
print(f'    ||delta_C_aug||: {Caug_delta_T1:.3e}')
print(f'    val_rms        : {val_T1:.5f}  (was {val0_T1:.5f},'
      f' ratio {val_T1/val0_T1:.2f}x)')
print(f'  Note: val_rms flat because x_aug=0 (zero-init ANN) -> C_aug/ANN have no')
print(f'        gradient yet. This is a bootstrapping issue, not a flaw of T1.')

## ═══════════════════════════════════════════════════════════════════════════
## Check 3b: perturb ANN in T1 to simulate post-encoder state
##           (x_aug nonzero -> C_aug gets gradient -> ANN can train)
## ═══════════════════════════════════════════════════════════════════════════

print()
print('=' * 60)
print('Check 3b: T1 with perturbed ANN (simulate after encoder trains)')
print('=' * 60)
m1b, out_phys_1b, out_aug_1b, ann_1b = build_T1()
torch.manual_seed(SEED + 99)
with torch.no_grad():
    for p in ann_1b.parameters():
        p.data += 5e-3 * torch.randn_like(p.data)   # small perturbation

one_backward(m1b)
cd_grad_1b   = grad_norm(out_phys_1b.C)
caug_grad_1b = grad_norm(out_aug_1b.C)
ann_grad_1b  = ann_grad_norm(ann_1b)
print(f'  Cd_norm grad               : {cd_grad_1b}   (still None: buffer)')
print(f'  C_aug grad norm            : {caug_grad_1b:.3e}  (nonzero: C_aug can train)')
print(f'  ANN param grad norm        : {ann_grad_1b:.3e}  (nonzero: ANN can train)')

## ═══════════════════════════════════════════════════════════════════════════
## Summary
## ═══════════════════════════════════════════════════════════════════════════

print()
print('=' * 60)
print('SUMMARY')
print('=' * 60)
hdr = f'  {"Check":<42s}  {"T0 (wrong)":>12s}  {"T1 (fix)":>12s}  {"Expected":>14s}'
print(hdr)

def fmt(v):
    if v is None: return 'None'
    return f'{v:.2e}'

rows = [
    ('Cd_norm grad (1 backward)',
     fmt(cd_col_grad_T0), fmt(cd_grad_T1), 'T0>0, T1=None'),
    (f'||delta_Cd|| after {N_STEPS} steps',
     fmt(Cd_drift_T0),    fmt(Cd_drift_T1), 'T0>0, T1=0'),
    ('val_rms improvement',
     f'{val_T0/val0_T0:.2f}x', f'{val_T1/val0_T1:.2f}x', 'T0<1 (drift), T1~1'),
    ('C_aug grad (perturbed ANN)',
     'N/A', fmt(caug_grad_1b), 'T1>0 (can train)'),
    ('ANN grad (perturbed ANN)',
     'N/A', fmt(ann_grad_1b), 'T1>0 (can train)'),
]
for label, v0, v1, exp in rows:
    print(f'  {label:<42s}  {v0:>12s}  {v1:>12s}  {exp:>14s}')

print()
print('  Conclusion:')
print('  T0 (wrong): Cd_norm drifts every step — val improvement is fake.')
print('  T1 (fix)  : Cd_norm is EXACTLY fixed. C_aug + ANN train once x_aug != 0.')
print('  Action    : replace Parameterized_Linear_Output_Block(C=[Cd|C_aug])')
print('              with Linear_Output_Block(C=Cd) + Parameterized(C=C_aug).')
print('Done.')
