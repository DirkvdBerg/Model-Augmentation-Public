"""
diag16_pole_routing_simulation.py
-----------------------------------
Synthetic verification that the blowup observed in gantry training is caused
by poles at |z|=1 and that our routing fix resolves it.

Uses the exact same block types and routing as gantry_interconnect_dynamic.py.
Gantry_State_Block is replaced by Linear_State_Block(A, B) with A chosen for
stable poles (k>0) or integrator poles (k=0). Everything else is identical.

Three cases:
  Case 1  Stable     + Config A (Jan's routing):  ANN->all xp,  y=C_phys@x_phys  -> no blowup
  Case 2  Integrator + Config A (Jan's routing):  ANN->all xp,  y=C_phys@x_phys  -> blowup
  Case 3  Integrator + Config B (our fix):         ANN->x_aug,   y=C_full@x       -> no blowup

Config A routing (lines 264 of gantry_interconnect_dynamic.py before D-065):
  ic.connect_block_signals(ann_block, ["x","u"], ["xp"])   # ANN to all xp

Config B routing (current state after D-065):
  ic.connect_block_signals(ann_block, ["x","u"], [])
  ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(AUG_IX, nxd))
  ic.connect_signals("x", out_block, "concat", selection_matrix(np.arange(nxd), nxd))
"""
import os
import sys
import time
import numpy as np
import scipy.linalg
import torch
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
    Linear_State_Block, Linear_Output_Block,
    Parameterized_Linear_Output_Block, Static_ANN_Block,
)
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug

## ═══════════════════════════════════════════════════════════════════════════
## Dimensions  (kept small for speed; structure identical to gantry)
## ═══════════════════════════════════════════════════════════════════════════

NX_PHYS = 2     # [q, dq]  — analogous to gantry NX_PHYS=6
NX_ANN  = 1     # one augmented state (absorber) — analogous to NX_ANN=2
nxd     = NX_PHYS + NX_ANN   # 3
nu      = 1
ny      = 2     # [position, velocity] — ny>1 required for deepSI apply_experiment
DT      = 0.005    # [s] — gantry uses 1/4000=0.00025 s; poles at |z|=1 regardless of DT
SEED    = 42
DTYPE_NP = np.float32
DTYPE_PT = torch.float32

NF         = 400     # rollout horizon — same as gantry
NA         = 4 * NX_PHYS + 1   # = 9  (Jan's formula: nxd*2+1 uses NX_PHYS here)
NB         = NA
NA_RIGHT   = 1
NB_RIGHT   = 1
N_LAYERS   = 2
N_NODES    = 16
LR         = 1e-4
BATCH_SIZE = 256
N_STEPS    = 20   # gradient steps (~1 epoch: 5000/256 ≈ 20 batches)

PHY_IX = np.arange(NX_PHYS)        # [0, 1]
AUG_IX = np.arange(NX_PHYS, nxd)   # [2]

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet',
                       'encoder-augmentation', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

## ═══════════════════════════════════════════════════════════════════════════
## System matrices
## ═══════════════════════════════════════════════════════════════════════════

# --- Stable system: critically damped spring-mass-damper ---
# k=100, c=20 -> omega_n=10, zeta=1.0 (critically damped)
# |z| = exp(-10*DT) = exp(-0.05) ≈ 0.951  ->  1-|z| ≈ 0.049  ->  BPTT amp ≈ 20x
# (vs unbounded for integrator) — makes stable vs integrator contrast clear
m_phys, k_phys, c_phys = 1.0, 100.0, 20.0
Ac_stable = np.array([[0.0, 1.0],
                      [-k_phys / m_phys, -c_phys / m_phys]])
Ad_stable = scipy.linalg.expm(Ac_stable * DT)
Bc_stable = np.array([[0.0], [1.0 / m_phys]])
Bd_stable = np.linalg.solve(Ac_stable, (Ad_stable - np.eye(2))) @ Bc_stable

# --- Integrator system: free mass (k=0, c=0) -> both poles at z=1 ---
# m*ddot_q = u  ->  A_c = [[0,1],[0,0]]  ->  A_d = [[1,DT],[0,1]]
Ad_int = np.array([[1.0, DT], [0.0, 1.0]])
Bd_int = np.array([[0.5 * DT**2 / m_phys], [DT / m_phys]])

# --- Physical output: position + velocity (ny=2; deepSI requires ny>1) ---
C_PHYS_TRUE = np.array([[1.0, 0.0],
                         [0.0, 1.0]], dtype=DTYPE_NP)   # (2, 2)
D_PHYS_TRUE = np.zeros((ny, nu), dtype=DTYPE_NP)

# --- Hidden absorber (same for both systems) ---
# 1D stable absorber driven by main mass position
# x_abs[k+1] = A_ABS * x_abs[k] + B_ABS * q[k]
# y_true[k]  = C_PHYS @ x_phys[k] + C_ABS * x_abs[k]
OMEGA_ABS  = 2.0 * np.pi * 2.0   # 2 Hz absorber
A_ABS      = float(np.exp(-OMEGA_ABS * DT))   # ~ 0.939
B_ABS      = 0.05    # absorber driven by main mass position q
C_ABS_TRUE_VEC = np.array([2.0, 0.0], dtype=DTYPE_NP)   # absorber -> position only (larger = more residual, closer to gantry NRMS)

# C_aug init for Config B (analogous to C_aug_init[2,0]=1e-2 in gantry)
C_AUG_INIT = np.array([[0.01], [0.0]], dtype=DTYPE_NP)   # (ny=2, NX_ANN=1)

## ═══════════════════════════════════════════════════════════════════════════
## Data generation
## ═══════════════════════════════════════════════════════════════════════════

N_TRAIN = 5000
N_VAL   = 2000

def simulate(A_d, B_d, u_seq):
    """Simulate physical + hidden absorber system. y includes absorber effect."""
    A_d = A_d.astype(DTYPE_NP)
    B_d = B_d.flatten().astype(DTYPE_NP)
    N   = len(u_seq)
    x_phys = np.zeros((N, NX_PHYS), dtype=DTYPE_NP)
    x_abs  = np.zeros(N, dtype=DTYPE_NP)
    y      = np.zeros((N, ny), dtype=DTYPE_NP)
    for k in range(N):
        y[k] = C_PHYS_TRUE @ x_phys[k] + C_ABS_TRUE_VEC * x_abs[k]
        if k < N - 1:
            x_phys[k+1] = A_d @ x_phys[k] + B_d * u_seq[k, 0]
            x_abs[k+1]  = A_ABS * x_abs[k] + B_ABS * x_phys[k, 0]
    return x_phys, y

rng = np.random.default_rng(SEED)
# Use lower amplitude for integrator to prevent state explosion
u_stable = rng.normal(0, 2.0, (N_TRAIN + N_VAL, nu)).astype(DTYPE_NP)
u_int    = rng.normal(0, 0.1, (N_TRAIN + N_VAL, nu)).astype(DTYPE_NP)

print('Generating synthetic data...')
systems_raw = {
    'stable':      (Ad_stable, Bd_stable, u_stable),
    'integrator':  (Ad_int,    Bd_int,    u_int),
}

DATA = {}
for name, (Ad, Bd, u_seq) in systems_raw.items():
    xp_all, y_all = simulate(Ad, Bd, u_seq)
    u_tr, y_tr, xp_tr = u_seq[:N_TRAIN], y_all[:N_TRAIN], xp_all[:N_TRAIN]
    u_va, y_va         = u_seq[N_TRAIN:], y_all[N_TRAIN:]

    train_sd = deepSI.System_data(u=u_tr, y=y_tr, dt=DT)
    val_sd   = deepSI.System_data(u=u_va, y=y_va, dt=DT)

    # Normalisation
    u_mean = u_tr.mean(axis=0).reshape(nu,     1).astype(DTYPE_NP)
    std_u  = u_tr.std(axis=0) .reshape(nu,     1).astype(DTYPE_NP) + 1e-8
    y0     = y_tr.mean(axis=0).astype(DTYPE_NP)
    ystd   = y_tr.std(axis=0) .astype(DTYPE_NP) + 1e-8
    x_mean = xp_tr.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
    std_x  = xp_tr.std(axis=0) .reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8

    C_phys_norm = C_PHYS_TRUE * std_x.flatten()[None, :] / ystd[:, None]  # (1, 2)

    sd_with_x = deepSI.System_data(u=u_tr, y=y_tr)
    sd_with_x.x = xp_tr
    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        Ad.astype(np.float64), Bd.astype(np.float64),
        C_PHYS_TRUE.astype(np.float64), D_PHYS_TRUE.astype(np.float64),
        sd_with_x,
    )

    DATA[name] = dict(
        Ad=Ad, Bd=Bd, Ad_bar=Ad_bar, Bd_bar=Bd_bar, Cd_bar=Cd_bar, Dd_bar=Dd_bar,
        train_sd=train_sd, val_sd=val_sd,
        u_mean=u_mean, std_u=std_u, y0=y0, ystd=ystd,
        x_mean=x_mean, std_x=std_x, C_phys_norm=C_phys_norm,
    )
    eigs = np.linalg.eigvals(Ad)
    print(f'  {name}: |z| = {np.abs(eigs)}  1-|z| = {1 - np.abs(eigs)}')

## ═══════════════════════════════════════════════════════════════════════════
## build_model — routing identical to gantry_interconnect_dynamic.py
## ═══════════════════════════════════════════════════════════════════════════

def build_model(system_name, config):
    """
    system_name : 'stable' | 'integrator'
    config      : 'A' (Jan's routing) | 'B' (our fix, current gantry_interconnect_dynamic.py)
    """
    d = DATA[system_name]
    np.random.seed(SEED); torch.manual_seed(SEED)

    ic = Interconnect(nxd, nu, ny, debugging=False)

    # Physics block: replaces Gantry_State_Block — same routing, different A/B
    phy_block = Linear_State_Block(
        A=torch.tensor(d['Ad_bar'], dtype=DTYPE_PT),
        B=torch.tensor(d['Bd_bar'], dtype=DTYPE_PT),
    )
    ic.add_block(phy_block)

    if config == 'A':
        # Jan's routing (gantry_interconnect_dynamic.py before D-065):
        # nw=nxd -> ANN outputs to all state rows
        ann_block = Static_ANN_Block(
            nz=nxd + nu, nw=nxd,
            n_nodes_per_layer=N_NODES, n_hidden_layers=N_LAYERS,
            net=zero_init_feed_forward_nn, activation=torch.nn.Tanh,
        )
        out_block = Linear_Output_Block(C=d['C_phys_norm'], D=D_PHYS_TRUE)
        ic.add_block(ann_block); ic.add_block(out_block)

        # ANN -> ALL xp rows (Jan's default — safe for stable systems, blowup for |z|=1)
        ic.connect_block_signals(ann_block, ["x", "u"], ["xp"])
        ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
        ic.connect_block_signals(phy_block, ["u"], [])
        ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
        ic.connect_signals("x", out_block, "concat", selection_matrix(PHY_IX, nxd))
        ic.connect_block_signals(out_block, ["u"], ["y"])

    elif config == 'B':
        # Our fix (gantry_interconnect_dynamic.py after D-065):
        # nw=NX_ANN -> ANN outputs only to augmented state rows
        ann_block = Static_ANN_Block(
            nz=nxd + nu, nw=NX_ANN,
            n_nodes_per_layer=N_NODES, n_hidden_layers=N_LAYERS,
            net=zero_init_feed_forward_nn, activation=torch.nn.Tanh,
        )
        C_full    = np.hstack([d['C_phys_norm'], C_AUG_INIT])   # (2, 3)
        out_block = Parameterized_Linear_Output_Block(
            C=C_full, D=D_PHYS_TRUE, flag_loss_reg=False)
        ic.add_block(ann_block); ic.add_block(out_block)

        # ANN -> x_aug rows only
        ic.connect_block_signals(ann_block, ["x", "u"], [])
        ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(AUG_IX, nxd))
        ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
        ic.connect_block_signals(phy_block, ["u"], [])
        ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
        # Output reads full x (physical + augmented)
        ic.connect_signals("x", out_block, "concat", selection_matrix(np.arange(nxd), nxd))
        ic.connect_block_signals(out_block, ["u"], ["y"])

    fit_sys = SSE_Interconnect(
        interconnect=ic, na=NA, nb=NB,
        na_right=NA_RIGHT, nb_right=NB_RIGHT,
        e_net_kwargs={"n_nodes_per_layer": N_NODES, "n_hidden_layers": N_LAYERS},
    )
    fit_sys.norm.u0   = d['u_mean'].flatten()
    fit_sys.norm.ustd = d['std_u'].flatten()
    fit_sys.norm.y0   = d['y0']
    fit_sys.norm.ystd = d['ystd']

    fit_sys.encoder = linear_encoder_init_aug(
        A=d['Ad_bar'], B=d['Bd_bar'], C=d['Cd_bar'], D=d['Dd_bar'],
        nx=NX_PHYS, nu=nu, ny=ny, na=NA, nb=NB,
        nx_aug=NX_ANN,
        n_nodes_per_layer=N_NODES, n_hidden_layers=N_LAYERS,
        flag_linear_only=False,
        u_mean=d['u_mean'], std_u=d['std_u'],
        y0=d['y0'], ystd=d['ystd'],
        x_mean=d['x_mean'], std_x=d['std_x'],
    ).to(DTYPE_PT)

    train_list = deepSI.System_data_list([d['train_sd']])
    fit_sys.init_model(sys_data=train_list, auto_fit_norm=False)
    fit_sys.hfn.to(DTYPE_PT)

    # Match lr to gantry training
    for pg in fit_sys.optimizer.param_groups:
        pg['lr'] = LR

    return fit_sys

## ═══════════════════════════════════════════════════════════════════════════
## Helpers
## ═══════════════════════════════════════════════════════════════════════════

def get_val_rms(m, system_name):
    m.eval()
    return float(m.cal_validation_error(DATA[system_name]['val_sd'],
                                        validation_measure='sim-RMS'))

def gradient_step(m, system_name, step=0):
    """One gradient step at nf=NF. Returns val sim-RMS after step."""
    train_list = deepSI.System_data_list([DATA[system_name]['train_sd']])
    td  = m.make_training_data(m.norm.transform(train_list), nf=NF)
    n   = len(td[0])
    idx = np.random.default_rng(SEED + step).choice(n, min(BATCH_SIZE, n), replace=False)
    batch = [torch.tensor(td[i][idx], dtype=DTYPE_PT) for i in range(len(td))]

    m.train()
    m.optimizer.zero_grad()
    loss = m.loss(*batch, nf=NF)
    if torch.isnan(loss) or torch.isinf(loss):
        return float('inf')
    loss.backward()
    m.optimizer.step()
    return get_val_rms(m, system_name)

## ═══════════════════════════════════════════════════════════════════════════
## Run the three cases
## ═══════════════════════════════════════════════════════════════════════════

CASES = [
    ('stable',     'A', 'Stable     + Config A (Jan)'),
    ('integrator', 'A', 'Integrator + Config A (Jan)'),
    ('integrator', 'B', 'Integrator + Config B (fix)'),
]

results   = {}
val_hists = {}

for sys_name, cfg, label in CASES:
    print(f'\n{"="*60}')
    print(f'{label}')
    print(f'{"="*60}')
    t0 = time.time()

    m          = build_model(sys_name, cfg)
    val_before = get_val_rms(m, sys_name)
    vals       = [val_before]

    for step in range(1, N_STEPS + 1):
        v = gradient_step(m, sys_name, step=step)
        vals.append(v)
        ratio = v / val_before if val_before > 0 else float('inf')
        flag  = 'BLOWUP' if ratio > 10 else ('DEGRADED' if ratio > 1.5 else ('worse' if ratio > 1.05 else 'stable'))
        print(f'  step {step}: val = {v:.5g}  x{ratio:.1f}  [{flag}]')
        if v == float('inf') or v > 100 * val_before:
            print('  (stopping early — numerical overflow)')
            vals += [float('inf')] * (N_STEPS - step)
            break

    ratio1 = vals[1] / val_before
    results[label]   = dict(val_before=val_before, vals=vals, ratio1=ratio1)
    val_hists[label] = vals
    print(f'  ({time.time()-t0:.1f}s)')

## ═══════════════════════════════════════════════════════════════════════════
## Summary table
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('SUMMARY  (val sim-RMS ratio vs init, max over 20 steps, nf=400)')
print('='*60)
print(f'  {"Case":<42s}  {"init val":>9s}  {"max ratio":>10s}  {"verdict":>9s}')
for _, _, label in CASES:
    r    = results[label]
    v0   = r['val_before']
    rmax = max(v / v0 for v in r['vals'] if v != float('inf') and v0 > 0)
    verdict  = 'BLOWUP' if rmax > 10 else ('DEGRADED' if rmax > 1.5 else ('worse' if rmax > 1.05 else 'stable'))
    rmax_str = f'{rmax:.1f}x' if rmax != float('inf') else 'inf'
    print(f'  {label:<42s}  {v0:9.5f}  {rmax_str:>10s}  {verdict:>9s}')

print()
print('  Expected:')
print('    Stable     + Config A : stable   (|z|=0.95, BPTT amp~20x, bounded degradation)')
print('    Integrator + Config A : DEGRADED (|z|=1.0,  BPTT amp unbounded -> significant degradation)')
print('    Integrator + Config B : stable   (ANN never touches physical state rows -> BPTT bounded)')

## ═══════════════════════════════════════════════════════════════════════════
## Plot
## ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 4))
colors = ['C0', 'C3', 'C2']
for (_, _, label), color in zip(CASES, colors):
    vals   = val_hists[label]
    v0     = vals[0]
    ratios = [min(v / v0, 1e4) for v in vals]   # cap for log scale
    ax.plot(range(len(ratios)), ratios, 'o-', color=color, label=label)

ax.axhline(1.0,  color='k',  lw=0.8, linestyle='--', label='No change')
ax.axhline(10.0, color='C3', lw=0.8, linestyle=':',  label='10x threshold')
ax.set_xlabel('Gradient step')
ax.set_ylabel('Val sim-RMS / init  (log scale)')
ax.set_title(f'Blowup comparison: poles at |z|=1 vs fix  (nf={NF})')
ax.set_yscale('log')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, 'diag16_pole_routing_simulation.png')
fig.savefig(out_path, dpi=150)
print(f'\nSaved: {out_path}')
print('Done.')
