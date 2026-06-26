"""
diag12_lpv_y_scheduling.py
--------------------------
Hypothesis: ANN->all-states blowup is caused by ANN perturbing x[2] (q3_logical =
Y-position), the LPV scheduling variable in Gantry_State_Block.deriv().

Y enters the forward pass as the argument of the rational inertia inverse:
    dY  = mh * (alpha*gamma - beta^2 + 2*beta*mh*Y + mh*(alpha-mh)*Y^2)   # det(M(Y))
    a   = N(Y)/dY @ fnet                                                    # accel
Any ANN perturbation to x[2] changes dY and the LPV dynamics every RK4 sub-step.

Two competing mechanisms:
  M1  Near-singularity: dY ~ 0 at nominal Y -> N/dY -> inf.
  M2  Large LPV Jacobian: dY is fine, but d/dY[N(Y)/dY] is large ->
      gradient at ANN output channel 2 is large -> weight update pushes Y hard
      -> simulation diverges on next forward pass.

Tests
-----
  S0  dY safety margin over val trajectory. If dY >> 0: M1 not the mechanism.

  T1  ANN -> all 8 states [0,1,2,3,4,5,6,7]  -- replicates diag9 blowup.
      Also reports gradient norm per ANN output channel before the weight update:
      if channel 2 >> others -> M2 confirmed.

  T2  ANN -> [0,1,3,4,5,6,7]  (exclude x[2] = Y-position, the scheduling var).
      Does excluding x[2] prevent blowup?

Falsifiable outcomes:
  T1 blows up AND T2 stable               -> hypothesis confirmed: x[2] is the cause.
  T1 blows up AND T2 also blows up        -> hypothesis wrong; x[2] is not the only cause.
  T1 stable (unexpected)                  -> diag9 not reproduced; check config.
  S0: dY >> 0 AND T1 ch2 grad >> others   -> mechanism is M2 (large LPV Jacobian).
  S0: dY ~ 0                              -> mechanism is M1 (singularity).

Runtime: ~2-4 min (1 training trajectory, 1 gradient step each).
Outputs: simulations/gantry_subnet/encoder-augmentation/diagnostics/diag12_*.{png,json}
"""

import os, sys, time, json
import numpy as np
import torch
from scipy.io import loadmat
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import deepSI
from model_augmentation.utils.utils import (
    normalize_linear_ss_matrices, expansion_matrix, selection_matrix,
)
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

SEED    = 42
NX_PHYS = 6
nu, ny  = 3, 3
NX_ANN  = 2
Y_OP    = None      # LPV self-scheduled

FS_ORIG  = 20000
FS_NEW   = 4000
D        = FS_ORIG // FS_NEW
TS_NEW   = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32

HP = dict(n_nodes_per_layer=16, n_hidden_layers=2, up_sample=2, batch_size=256, lr=1e-4)
NF_DIAG = max(1, int(0.100 / TS_NEW))   # 400 samples -- matches main training script

## ═══════════════════════════════════════════════════════════════════════════
## Data
## ═══════════════════════════════════════════════════════════════════════════

TRAJ_DIR = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab', 'multisine')

def _load_u(d):
    return d['u_total'] if 'u_total' in d else d['u']

def load_traj(f):
    d = loadmat(os.path.join(TRAJ_DIR, f), squeeze_me=True)
    return deepSI.System_data(
        u=_load_u(d)[::D].astype(DTYPE_NP),
        y=d['y'][::D].astype(DTYPE_NP),
        dt=TS_NEW,
    )

print('Loading data ...')
train_list = [load_traj('T1_Y_sweep_conservative.mat')]
train_data = deepSI.System_data_list(train_list)
val_data   = load_traj('V1_X_sym_Y_mid_sweep.mat')
print(f'  train: {len(train_list[0].y)} samples | val: {len(val_data.y)} samples')

## ═══════════════════════════════════════════════════════════════════════════
## Normalisation  (mirrors gantry_interconnect_dynamic.py)
## ═══════════════════════════════════════════════════════════════════════════

u_all = np.concatenate([t.u for t in train_list])
y_all = np.concatenate([t.y for t in train_list])
P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)

x_logical_list = []
for t in train_list:
    pos = (P_inv_T @ t.y.T).T
    vel = np.diff(pos, axis=0) * (1.0 / t.dt)
    vel = np.vstack([vel[:1], vel])
    x_logical_list.append(np.hstack([pos, vel]))
x_all = np.concatenate(x_logical_list)

x_mean  = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
std_x   = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
std_u   = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
u_mean  = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
ystd    = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
y0      = y_all.mean(axis=0).astype(DTYPE_NP)
Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
Dd_np   = Dd.numpy()
PHY_IX  = np.arange(NX_PHYS)
nxd     = NX_PHYS + NX_ANN

## ═══════════════════════════════════════════════════════════════════════════
## Output directory
## ═══════════════════════════════════════════════════════════════════════════

run_id   = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet',
                        'encoder-augmentation', 'diagnostics')
os.makedirs(save_dir, exist_ok=True)

## ═══════════════════════════════════════════════════════════════════════════
## build_variant(ann_ix)
## ═══════════════════════════════════════════════════════════════════════════

def build_variant(ann_ix):
    """Build SSE_Interconnect with ANN routed to state indices ann_ix.

    ann_ix  1-D int array of state indices the ANN updates.
            Physical block always receives/updates PHY_IX=[0..5].
            Output block always reads PHY_IX.
    Returns (fit_sys, ann_block).
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ic = Interconnect(nxd, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
        up_sample=HP['up_sample'],
    ).to(DTYPE_PT)
    out_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    ic.add_block(phy_block)
    ic.add_block(out_block)

    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=len(ann_ix),
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=torch.nn.Tanh,
    )
    ic.add_block(ann_block)

    ic.connect_block_signals(ann_block, ["x", "u"], [])
    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(ann_ix, nxd))
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", out_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_block, ["u"], ["y"])

    # HEURISTIC: na = 4*NX_PHYS+1 (Jan's rule of thumb); na_right=1 for linear_map encoder
    na = 4 * NX_PHYS + 1
    nb = na
    fit_sys = SSE_Interconnect(
        interconnect=ic, na=na, nb=nb, na_right=1, nb_right=1,
        e_net_kwargs={
            "n_nodes_per_layer": HP['n_nodes_per_layer'],
            "n_hidden_layers":   HP['n_hidden_layers'],
        },
    )
    fit_sys.norm.u0   = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0   = y0
    fit_sys.norm.ystd = ystd

    # Encoder: use finite-diff states (single T1 trajectory -- baseline_states.npz has all 8)
    Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)
    sd   = deepSI.System_data(u=u_all, y=y_all)
    sd.x = x_all   # finite-diff physical states from T1
    Ad_b, Bd_b, Cd_b, Dd_b = normalize_linear_ss_matrices(Ad, Bd, Cd_dt, Dd_dt, sd)
    fit_sys.encoder = linear_encoder_init_aug(
        A=Ad_b, B=Bd_b, C=Cd_b, D=Dd_b,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb, nx_aug=NX_ANN,
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        flag_linear_only=False,
        u_mean=u_mean, std_u=std_u, y0=y0, ystd=ystd,
        x_mean=x_mean, std_x=std_x,
    ).to(DTYPE_PT)

    fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
    fit_sys.hfn.to(DTYPE_PT)
    return fit_sys, ann_block


## ═══════════════════════════════════════════════════════════════════════════
## S0: dY safety margin  (no training)
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('S0: dY = det(M(Y)) safety margin')
print('='*60)

# Y = q3_logical from validation trajectory
q_logical_val = (P_inv_T @ val_data.y.T).T   # (N, 3) logical coordinates
Y_vals = q_logical_val[:, 2]                  # q3 = Y-direction position [m]

# Read LFR physical constants from a reference block (no training needed)
_ref = Gantry_State_Block(Y_op=None, std_x=std_x, std_u=std_u,
                           x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW)
mh_v = float(_ref.mh);    a_v = float(_ref.alpha)
b_v  = float(_ref.beta);  g_v = float(_ref.gamma_)

# THEORY: dY = det(M(Y)) from LFR rational structure (Gantry_State_Block.deriv lines 799-801)
dY_vals = mh_v * (a_v*g_v - b_v**2
                  + 2*b_v*mh_v*Y_vals
                  + mh_v*(a_v - mh_v)*Y_vals**2)

# Roots of dY polynomial: mh*(a-mh)*Y^2 + 2*b*mh*Y + (a*g - b^2) = 0
# THEORY: quadratic formula applied to det(M(Y)) = 0 (Gantry_State_Block.deriv)
poly_coeffs = [mh_v*(a_v - mh_v), 2*b_v*mh_v, a_v*g_v - b_v**2]
roots_dY = np.roots(poly_coeffs)

print(f'  Y range (val):        [{Y_vals.min():.4f}, {Y_vals.max():.4f}] m')
print(f'  dY range (val):       [{dY_vals.min():.4e}, {dY_vals.max():.4e}]')
print(f'  dY polynomial roots:  {roots_dY}')
real_roots = np.real(roots_dY[np.isreal(roots_dY)])
if len(real_roots) > 0:
    min_dist = float(np.abs(Y_vals[:, None] - real_roots[None, :]).min())
    print(f'  Min |Y_val - nearest root|: {min_dist:.4f} m')
    if min_dist < 0.05:
        print('  WARNING: nominal Y is close (<0.05 m) to singularity -- M1 possible.')
    else:
        print('  dY >> 0 at nominal Y.  M1 (singularity) is NOT the mechanism.')
        print('  If T1 still blows up -> investigate M2 (large LPV Jacobian at x[2]).')
else:
    print('  No real roots (M(Y) positive-definite for all real Y). M1 ruled out.')

## ═══════════════════════════════════════════════════════════════════════════
## T1: ANN -> all 8 states
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('T1: ANN -> all 8 states [0,1,2,3,4,5,6,7]')
print('='*60)
t0 = time.time()

ANN_IX_T1 = np.arange(nxd)            # [0,1,2,3,4,5,6,7]
m1, ann1  = build_variant(ANN_IX_T1)
val_T1_before = float(m1.cal_validation_error(val_data, validation_measure='sim-RMS'))
print(f'  val before: {val_T1_before:.5f}')

# Single backward pass: measure gradient norm per ANN output channel
rng1   = np.random.default_rng(SEED)
data1  = m1.make_training_data(m1.norm.transform(train_data), nf=NF_DIAG)
idx1   = rng1.choice(len(data1[0]), HP['batch_size'], replace=False)
batch1 = [torch.tensor(data1[i][idx1], dtype=DTYPE_PT) for i in range(len(data1))]

m1.train()
m1.optimizer.zero_grad()
loss1 = m1.loss(*batch1, nf=NF_DIAG)
loss1.backward()

# Final layer weight gradient: shape (nw, n_hidden). Row i -> ANN output channel i.
# ANN output channel i maps to global state index ANN_IX_T1[i].
final_T1  = ann1.net.net[-1]
W_grad_T1 = final_T1.weight.grad    # (nw=8, n_hidden=16)
grad_per_ch_T1 = {}
STATE_NAMES = ['q1', 'q2', 'q3(Y)', 'dq1', 'dq2', 'dq3', 'x_aug0', 'x_aug1']
print('  Gradient norm per ANN output channel (before weight update):')
if W_grad_T1 is not None:
    for i, si in enumerate(ANN_IX_T1):
        g = float(W_grad_T1[i].norm().item())
        grad_per_ch_T1[int(si)] = g
        marker = '  <-- LPV scheduling var' if si == 2 else ''
        print(f'    x[{si}] {STATE_NAMES[si]:10s}  {g:.3e}{marker}')
else:
    print('    (no gradient -- ANN disconnected from loss)')

# Actual weight update, then measure val sim-RMS
m1.optimizer.step()
val_T1_after = float(m1.cal_validation_error(val_data, validation_measure='sim-RMS'))
ratio_T1 = val_T1_after / max(val_T1_before, 1e-10)
flag_T1  = 'BLOWUP' if ratio_T1 > 5 else ('worse' if ratio_T1 > 1.05 else 'stable')
print(f'\n  val after:  {val_T1_after:.5f}  [{flag_T1}]  (x{ratio_T1:.1f})')
print(f'  ({time.time()-t0:.0f}s)')

## ═══════════════════════════════════════════════════════════════════════════
## T2: ANN -> [0,1,3,4,5,6,7]  (exclude x[2] = Y-position)
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('T2: ANN -> [0,1,3,4,5,6,7]  (exclude x[2] = Y-position)')
print('='*60)
t0 = time.time()

ANN_IX_T2 = np.array([0, 1, 3, 4, 5, 6, 7])   # x[2] excluded
m2, ann2  = build_variant(ANN_IX_T2)
val_T2_before = float(m2.cal_validation_error(val_data, validation_measure='sim-RMS'))
print(f'  val before: {val_T2_before:.5f}')

rng2   = np.random.default_rng(SEED)
data2  = m2.make_training_data(m2.norm.transform(train_data), nf=NF_DIAG)
idx2   = rng2.choice(len(data2[0]), HP['batch_size'], replace=False)
batch2 = [torch.tensor(data2[i][idx2], dtype=DTYPE_PT) for i in range(len(data2))]

m2.train()
m2.optimizer.zero_grad()
loss2 = m2.loss(*batch2, nf=NF_DIAG)
loss2.backward()

final_T2  = ann2.net.net[-1]
W_grad_T2 = final_T2.weight.grad    # (nw=7, n_hidden=16)
grad_per_ch_T2 = {}
print('  Gradient norm per ANN output channel (before weight update):')
if W_grad_T2 is not None:
    for i, si in enumerate(ANN_IX_T2):
        g = float(W_grad_T2[i].norm().item())
        grad_per_ch_T2[int(si)] = g
        print(f'    x[{si}] {STATE_NAMES[si]:10s}  {g:.3e}')
else:
    print('    (no gradient -- ANN disconnected from loss)')

m2.optimizer.step()
val_T2_after = float(m2.cal_validation_error(val_data, validation_measure='sim-RMS'))
ratio_T2 = val_T2_after / max(val_T2_before, 1e-10)
flag_T2  = 'BLOWUP' if ratio_T2 > 5 else ('worse' if ratio_T2 > 1.05 else 'stable')
print(f'\n  val after:  {val_T2_after:.5f}  [{flag_T2}]  (x{ratio_T2:.1f})')
print(f'  ({time.time()-t0:.0f}s)')

## ═══════════════════════════════════════════════════════════════════════════
## T3: ANN -> [2] only  (just Y-position -- isolation test)
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('T3: ANN -> [2] only  (Y-position in isolation)')
print('='*60)
t0 = time.time()

ANN_IX_T3 = np.array([2])              # x[2] = q3_logical = Y-position only
m3, ann3  = build_variant(ANN_IX_T3)
val_T3_before = float(m3.cal_validation_error(val_data, validation_measure='sim-RMS'))
print(f'  val before: {val_T3_before:.5f}')

rng3   = np.random.default_rng(SEED)
data3  = m3.make_training_data(m3.norm.transform(train_data), nf=NF_DIAG)
idx3   = rng3.choice(len(data3[0]), HP['batch_size'], replace=False)
batch3 = [torch.tensor(data3[i][idx3], dtype=DTYPE_PT) for i in range(len(data3))]

m3.train()
m3.optimizer.zero_grad()
loss3 = m3.loss(*batch3, nf=NF_DIAG)
loss3.backward()

final_T3  = ann3.net.net[-1]
W_grad_T3 = final_T3.weight.grad    # (nw=1, n_hidden=16)
grad_per_ch_T3 = {}
if W_grad_T3 is not None:
    g = float(W_grad_T3[0].norm().item())
    grad_per_ch_T3[2] = g
    print(f'  Gradient norm: x[2] q3(Y): {g:.3e}')
else:
    print('  (no gradient)')

m3.optimizer.step()
val_T3_after = float(m3.cal_validation_error(val_data, validation_measure='sim-RMS'))
ratio_T3 = val_T3_after / max(val_T3_before, 1e-10)
flag_T3  = 'BLOWUP' if ratio_T3 > 5 else ('worse' if ratio_T3 > 1.05 else 'stable')
print(f'\n  val after:  {val_T3_after:.5f}  [{flag_T3}]  (x{ratio_T3:.1f})')
print(f'  ({time.time()-t0:.0f}s)')

## ═══════════════════════════════════════════════════════════════════════════
## Summary
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('Summary')
print('='*60)
print(f'  {"Test":<6}  {"ANN routing":<32}  {"Before":>8}  {"After":>8}  Verdict')
print(f'  {"T1":<6}  {"all [0..7]":<32}  {val_T1_before:>8.5f}  {val_T1_after:>8.5f}  {flag_T1}')
print(f'  {"T2":<6}  {"[0,1,3,4,5,6,7] (no x[2])":<32}  {val_T2_before:>8.5f}  {val_T2_after:>8.5f}  {flag_T2}')
print(f'  {"T3":<6}  {"[2] only":<32}  {val_T3_before:>8.5f}  {val_T3_after:>8.5f}  {flag_T3}')
print()
t1_bad = flag_T1 in ('BLOWUP', 'worse')
t2_ok  = flag_T2 == 'stable'
t3_bad = flag_T3 in ('BLOWUP', 'worse')
if t1_bad and t2_ok and t3_bad:
    print('  HYPOTHESIS CONFIRMED: x[2] alone causes blowup; excluding x[2] prevents it.')
    print('  x[2] (Y-position) is the LPV scheduling var and its perturbation destabilises')
    print('  Gantry_State_Block.deriv() via dY = det(M(Y)).')
    print('  Next: run full training with ANN_IX = [0,1,3,4,5,6,7].')
elif t1_bad and t2_ok and not t3_bad:
    print('  PARTIAL: T2 stable (excluding x[2] helps) but T3 also stable (x[2] alone safe).')
    print('  x[2] is not sufficient on its own -- blowup requires x[2] + other states.')
    print('  Investigate interaction; consider excluding x[2] as a necessary but not sufficient fix.')
elif t1_bad and not t2_ok:
    print('  HYPOTHESIS NOT CONFIRMED: T2 also degrades.')
    print('  x[2] is not the sole cause. Investigate x[0], x[1], x[3..5] individually.')
else:
    print('  Unexpected result. Inspect raw values and check initial val_before.')
    print('  If initial RMS is already high, the blowup signal may be obscured.')

## ═══════════════════════════════════════════════════════════════════════════
## Plots
## ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# -- S0: dY over validation trajectory --
ax = axes[0]
t_v = np.arange(len(Y_vals)) * TS_NEW
ax.plot(t_v, dY_vals, 'C0', lw=0.8, label='dY = det(M(Y))')
ax.axhline(0.0, color='k', lw=0.8, linestyle='--', alpha=0.6, label='dY = 0 (singularity)')
ax.set_xlabel('Time [s]')
ax.set_ylabel('dY = det(M(Y))')
ax.set_title('S0: LPV denominator over val trajectory\n'
             'If dY >> 0: singularity (M1) is not the blowup mechanism')
ax.legend(fontsize=8)
ax.grid(True)

# -- T1: gradient norm per output channel --
ax = axes[1]
if grad_per_ch_T1:
    ch_list = sorted(grad_per_ch_T1.keys())
    gnorms  = [grad_per_ch_T1[c] for c in ch_list]
    colors  = ['C3' if c == 2 else 'C0' for c in ch_list]
    ax.bar(ch_list, gnorms, color=colors, alpha=0.85)
    ax.set_xlabel('State index (ANN output channel)')
    ax.set_ylabel('||∂loss/∂W_ann_final|| per row')
    ax.set_title('T1 gradient per output channel\n'
                 'Red bar = x[2] = Y-position (LPV var)')
    ax.set_xticks(ch_list)
    ax.set_xticklabels([STATE_NAMES[c] for c in ch_list], rotation=45, ha='right', fontsize=7)
    ax.grid(True, axis='y')

# -- Before / after sim-RMS for T1, T2, T3 --
ax = axes[2]
tests  = ['T1\n(all)', 'T2\n(no x[2])', 'T3\n(x[2] only)']
before = [val_T1_before, val_T2_before, val_T3_before]
after  = [val_T1_after,  val_T2_after,  val_T3_after]
xp     = np.arange(3)
bw     = 0.3
ax.bar(xp - bw/2, before, bw, label='Before (0 steps)', color='C0', alpha=0.85)
ax.bar(xp + bw/2, after,  bw, label='After (1 step)',   color='C1', alpha=0.85)
ax.set_xticks(xp)
ax.set_xticklabels(tests)
ax.set_ylabel('Val sim-RMS')
ax.set_title('Isolation test\n'
             'T1 up + T2 stable + T3 up = x[2] isolated as cause')
ax.legend(fontsize=8)
ax.grid(True, axis='y')

fig.tight_layout()
fig_path = os.path.join(save_dir, f'diag12_lpv_y_scheduling_{run_id}.png')
fig.savefig(fig_path, dpi=150)
plt.close('all')
print(f'\nSaved figure: {fig_path}')

## ═══════════════════════════════════════════════════════════════════════════
## Save JSON
## ═══════════════════════════════════════════════════════════════════════════

results = dict(
    val_T1_before=val_T1_before, val_T1_after=val_T1_after,
    ratio_T1=ratio_T1,           verdict_T1=flag_T1,
    val_T2_before=val_T2_before, val_T2_after=val_T2_after,
    ratio_T2=ratio_T2,           verdict_T2=flag_T2,
    val_T3_before=val_T3_before, val_T3_after=val_T3_after,
    ratio_T3=ratio_T3,           verdict_T3=flag_T3,
    dY_min=float(dY_vals.min()), dY_max=float(dY_vals.max()),
    Y_min=float(Y_vals.min()),   Y_max=float(Y_vals.max()),
    dY_roots=[{'real': float(np.real(r)), 'imag': float(np.imag(r))} for r in roots_dY],
    ANN_IX_T1=ANN_IX_T1.tolist(), ANN_IX_T2=ANN_IX_T2.tolist(),
    ANN_IX_T3=ANN_IX_T3.tolist(),
    grad_per_ch_T1=grad_per_ch_T1, grad_per_ch_T2=grad_per_ch_T2,
    grad_per_ch_T3=grad_per_ch_T3,
    NF_DIAG=NF_DIAG, FS_NEW=FS_NEW, NX_ANN=NX_ANN,
)
json_path = os.path.join(save_dir, f'diag12_lpv_y_scheduling_{run_id}.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved JSON:   {json_path}')
