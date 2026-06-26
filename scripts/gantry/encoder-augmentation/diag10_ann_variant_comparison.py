"""
diag10_ann_variant_comparison.py
---------------------------------
Compares three ANN variants for the augmented gantry model to diagnose
the zero-gradient cascade problem found in diag9.

Background
----------
diag9 showed that training at nf=400 immediately blows up: after 1 step
the ANN hfn_grad jumps from 0.259 to 356.  The root cause is a gradient
cascade in zero_init_feed_forward_nn:

  * At init: W3 = 0  ->  dL/dW1 = dL/dout * W3^T * ... = 0.
  * Step 1 makes W3 non-zero.
  * Step 2: W1/W2 suddenly receive full gradient -> gnorm_hfn explodes.

Three variants are compared:
  A  (current)  zero_init_feed_forward_nn + Tanh     -- cascade expected
  B  (ECC-style) zero_init_feed_forward_nn + Identity -- cascade expected
  C  (journal)   zero_init_linear_mapping              -- no cascade

Tests run per variant:
  TA  Blowup test        : 2 gradient steps at nf=400, val before/after
  TB  Gradient cascade   : 1 backward pass at nf=2, per-layer gnorms
  TC  ANN-only blowup    : freeze encoder, 2 steps at nf=400
  TD  Grad norm vs nf    : gnorm_hfn for nf in [2, 5, 25, 100, 400]

All tests use a fresh model (same seed) per variant.
Results saved to scripts/gantry/encoder-augmentation/diagnostics/

Usage:
    conda run -n GraduationProject python \\
        scripts/gantry/encoder-augmentation/diag10_ann_variant_comparison.py
"""

import os
import sys
import time
import json
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
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn, zero_init_linear_mapping
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

N_BLOWUP_STEPS = 2        # TA/TC: gradient steps to run
NF_BLOWUP      = 400      # TA/TC: rollout length for blowup test
NF_CASCADE     = 2        # TB: minimum nf for gradient to reach ANN weights
NF_SWEEP_TD    = [2, 5, 25, 100, 400]  # TD: nf values to sweep
N_TRAIN_TRAJ   = 1        # how many training trajectories to load (1=fast, 8=full)

SAVE_DIR = os.path.join(SCRIPT_DIR, 'diagnostics')
os.makedirs(SAVE_DIR, exist_ok=True)

## ═══════════════════════════════════════════════════════════════════════════
## Model constants (identical to diag9 / gantry_interconnect_dynamic.py)
## ═══════════════════════════════════════════════════════════════════════════

NX_PHYS  = 6
nu       = 3
ny       = 3
Y_OP     = None
SEED     = 42

FS_ORIG  = 20000
FS_NEW   = 4000
D        = FS_ORIG // FS_NEW
TS_NEW   = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32

DEFAULT_HP = dict(
    NX_ANN=2,
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=2,
    nf=max(1, int(0.100 / TS_NEW)),   # 400
    na_nb=0,
    batch_size=256,
    lr=1e-4,
    epochs=10,
)
DEFAULT_HP['na_nb'] = (NX_PHYS + DEFAULT_HP['NX_ANN']) * 2 + 1

## ═══════════════════════════════════════════════════════════════════════════
## Data loading
## ═══════════════════════════════════════════════════════════════════════════

DATA_SUBDIR = 'multisine'
TRAJ_DIR    = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab', DATA_SUBDIR)

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

print('Loading data...')
train_list = [load_traj(f) for f in TRAIN_FILES[:N_TRAIN_TRAJ]]
train_data = deepSI.System_data_list(train_list)
val_data   = load_traj(VAL_FILE)
_n_val     = len(val_data.y)
print(f'  {len(train_list)} train trajectories | val: {_n_val} samples')

## ═══════════════════════════════════════════════════════════════════════════
## Normalisation
## ═══════════════════════════════════════════════════════════════════════════

u_all   = np.concatenate([t.u for t in train_list])
y_all   = np.concatenate([t.y for t in train_list])
fs      = 1.0 / train_list[0].dt
P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)

x_logical_list = []
for t in train_list:
    pos = (P_inv_T @ t.y.T).T
    vel = np.diff(pos, axis=0) * fs
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

## ═══════════════════════════════════════════════════════════════════════════
## Model builder
## ═══════════════════════════════════════════════════════════════════════════

def _get_encoder_dims(hp):
    na = 4 * NX_PHYS + 1
    return na, na, 1, 1

def _build_model(hp, ann_net_cls, ann_activation):
    """Build SSE_Interconnect with specified ANN net class and activation."""
    NX_ANN = hp['NX_ANN']
    nxd    = NX_PHYS + NX_ANN
    na, nb, na_right, nb_right = _get_encoder_dims(hp)

    ic = Interconnect(nxd, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
        up_sample=hp['up_sample'],
    ).to(DTYPE_PT)
    out_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    ic.add_block(phy_block)
    ic.add_block(out_block)

    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=nxd,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=ann_net_cls,
        activation=ann_activation,
    )
    ic.add_block(ann_block)

    ic.connect_block_signals(ann_block, ["x", "u"], ["xp"])
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
    if os.path.exists(baseline_npz):
        bl = np.load(baseline_npz, allow_pickle=True)
        x_phys_all = np.concatenate(bl['x_train_phys'])
    else:
        x_phys_all = x_all
    sys_data_with_x = deepSI.System_data(u=u_all, y=y_all)
    sys_data_with_x.x = x_phys_all
    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        Ad, Bd, Cd_dt, Dd_dt, sys_data_with_x)
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

VARIANT_DEFS = {
    'A_ffnn_tanh':     (zero_init_feed_forward_nn, torch.nn.Tanh),
    'B_ffnn_identity': (zero_init_feed_forward_nn, torch.nn.Identity),
    'C_linear':        (zero_init_linear_mapping,  torch.nn.Identity),
}
VARIANT_LABELS = {
    'A_ffnn_tanh':     'A: FFNN+Tanh (current)',
    'B_ffnn_identity': 'B: FFNN+Identity (ECC)',
    'C_linear':        'C: Linear (journal)',
}

def build_fresh(variant_key):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    net_cls, act_cls = VARIANT_DEFS[variant_key]
    return _build_model(DEFAULT_HP, net_cls, act_cls)

## ═══════════════════════════════════════════════════════════════════════════
## Shared helpers (reused from diag9)
## ═══════════════════════════════════════════════════════════════════════════

def make_data(fit_sys, nf):
    return fit_sys.make_training_data(fit_sys.norm.transform(train_data), nf=nf)

def manual_step(fit_sys, data_train, nf, rng):
    """One mini-batch gradient step.
    Returns (sqrt_train_loss, grad_norm_encoder, grad_norm_hfn).
    """
    n_total = len(data_train[0])
    idx     = rng.choice(n_total, DEFAULT_HP['batch_size'], replace=False)
    batch   = [torch.tensor(d[idx], dtype=DTYPE_PT) for d in data_train]

    fit_sys.train()
    fit_sys.optimizer.zero_grad()
    loss_val = fit_sys.loss(*batch, nf=nf)
    loss_val.backward()

    def _gnorm(module):
        sq = sum(
            p.grad.detach().norm().item() ** 2
            for p in module.parameters() if p.grad is not None
        )
        return float(sq ** 0.5)

    gnorm_enc = _gnorm(fit_sys.encoder)
    gnorm_hfn = _gnorm(fit_sys.hfn)
    fit_sys.optimizer.step()
    return float(loss_val.item()) ** 0.5, gnorm_enc, gnorm_hfn

def get_val_sim_rms(fit_sys):
    fit_sys.eval()
    return float(fit_sys.cal_validation_error(val_data, validation_measure='sim-RMS'))

def _get_ann_block(fit_sys):
    return next(b for b in fit_sys.hfn.connected_blocks if isinstance(b, Static_ANN_Block))

def _get_ann_layer_gnorms(ann_blk, variant_key):
    """Return list of (name, gnorm, shape) for all ANN parameters.
    Works for both FFNN variants (net.net) and linear variant (net_lin).
    """
    net = ann_blk.net
    if hasattr(net, 'net'):
        # zero_init_feed_forward_nn: ann_blk.net.net is nn.Sequential
        named = list(net.net.named_parameters())
    else:
        # zero_init_linear_mapping: ann_blk.net.net_lin is nn.Linear
        named = list(net.net_lin.named_parameters())

    result = []
    for name, p in named:
        if p.grad is not None:
            result.append((name, p.grad.norm().item(), list(p.shape)))
        else:
            result.append((name, 0.0, list(p.shape)))
    return result

def _get_final_weight_grad(ann_blk, variant_key):
    """Return grad of the final linear layer weight, zeros if None."""
    NX_ANN = DEFAULT_HP['NX_ANN']
    nxd    = NX_PHYS + NX_ANN
    net    = ann_blk.net
    if hasattr(net, 'net'):
        final = net.net[-1]
    else:
        final = net.net_lin
    W_grad = final.weight.grad
    if W_grad is None:
        W_grad = torch.zeros(nxd, final.weight.shape[1])
    return W_grad

## ═══════════════════════════════════════════════════════════════════════════
## Test functions
## ═══════════════════════════════════════════════════════════════════════════

def run_TA_blowup(variant_key):
    """TA: 2 gradient steps at nf=400. Returns dict with val_before, val_after,
    gnorm_hfn per step, sqrt_loss per step."""
    rng = np.random.default_rng(SEED)
    m   = build_fresh(variant_key)
    data = make_data(m, NF_BLOWUP)

    val_before = get_val_sim_rms(m)
    vals = [val_before]
    gnorms_hfn = []
    gnorms_enc = []
    losses     = []

    for _ in range(N_BLOWUP_STEPS):
        sl, ge, gh = manual_step(m, data, NF_BLOWUP, rng)
        vals.append(get_val_sim_rms(m))
        gnorms_hfn.append(gh)
        gnorms_enc.append(ge)
        losses.append(sl)

    return dict(vals=vals, gnorms_hfn=gnorms_hfn, gnorms_enc=gnorms_enc,
                losses=losses, val_before=val_before, val_after=vals[-1])

def run_TB_cascade(variant_key):
    """TB: 1 backward pass at nf=NF_CASCADE (=2). Returns per-layer gnorms
    and grad_phys / grad_aug split of the final layer."""
    NX_ANN = DEFAULT_HP['NX_ANN']
    nxd    = NX_PHYS + NX_ANN

    rng  = np.random.default_rng(SEED)
    m    = build_fresh(variant_key)
    data = make_data(m, NF_CASCADE)

    n_total = len(data[0])
    idx     = rng.choice(n_total, DEFAULT_HP['batch_size'], replace=False)
    batch   = [torch.tensor(data[i][idx], dtype=DTYPE_PT) for i in range(len(data))]

    m.train()
    m.optimizer.zero_grad()
    loss = m.loss(*batch, nf=NF_CASCADE)
    loss.backward()

    ann_blk = _get_ann_block(m)
    layer_gnorms = _get_ann_layer_gnorms(ann_blk, variant_key)
    W_grad = _get_final_weight_grad(ann_blk, variant_key)

    grad_phys = W_grad[:NX_PHYS, :].norm().item()
    grad_aug  = W_grad[NX_PHYS:nxd, :].norm().item()
    per_row   = W_grad.norm(dim=1).detach().numpy()

    # Are hidden layers getting gradient? (only meaningful for FFNN variants)
    # For FFNN: layer_gnorms[-2] is the final weight ('4.weight'); anything before is hidden.
    # For linear: no hidden layers — report None (cascade does not apply).
    net = ann_blk.net
    if hasattr(net, 'net'):
        final_weight_name = layer_gnorms[-2][0]   # e.g. '4.weight'
        hidden_grads_zero = all(
            g == 0.0 for name, g, _ in layer_gnorms
            if 'weight' in name and name != final_weight_name
        )
    else:
        hidden_grads_zero = None  # linear variant: no hidden layers

    return dict(
        layer_gnorms=layer_gnorms,
        grad_phys=grad_phys,
        grad_aug=grad_aug,
        per_row=per_row,
        hidden_grads_zero=hidden_grads_zero,
        sqrt_loss=float(loss.item()) ** 0.5,
    )

def run_TC_ann_only_blowup(variant_key):
    """TC: Freeze encoder, train ANN only for 2 steps at nf=400.
    Returns val trajectory and gnorm_hfn per step."""
    rng  = np.random.default_rng(SEED + 120)
    m    = build_fresh(variant_key)
    for p in m.encoder.parameters():
        p.requires_grad_(False)

    data = make_data(m, NF_BLOWUP)
    vals = [get_val_sim_rms(m)]
    gnorms_hfn = []
    gnorms_enc = []

    for _ in range(N_BLOWUP_STEPS):
        sl, ge, gh = manual_step(m, data, NF_BLOWUP, rng)
        vals.append(get_val_sim_rms(m))
        gnorms_hfn.append(gh)
        gnorms_enc.append(ge)

    return dict(vals=vals, gnorms_hfn=gnorms_hfn, gnorms_enc=gnorms_enc)

def run_TD_grad_vs_nf(variant_key):
    """TD: For each nf in NF_SWEEP_TD, build fresh model, take 1 step, record
    gnorm_hfn and val sim-RMS after. Returns dicts keyed by nf."""
    gnorms_hfn = {}
    gnorms_enc = {}
    vals_after = {}

    for nf in NF_SWEEP_TD:
        rng = np.random.default_rng(SEED + 140)
        m   = build_fresh(variant_key)
        data = make_data(m, nf)
        sl, ge, gh = manual_step(m, data, nf, rng)
        vals_after[nf] = get_val_sim_rms(m)
        gnorms_hfn[nf] = gh
        gnorms_enc[nf] = ge

    return dict(gnorms_hfn=gnorms_hfn, gnorms_enc=gnorms_enc, vals_after=vals_after)

## ═══════════════════════════════════════════════════════════════════════════
## Run all tests for all variants
## ═══════════════════════════════════════════════════════════════════════════

_t_script_start = time.time()
results = {}

for vk in VARIANT_DEFS:
    lbl = VARIANT_LABELS[vk]
    print(f'\n{"="*70}')
    print(f'Variant {lbl}')
    print(f'{"="*70}')

    t0 = time.time()
    print(f'  [TA] blowup ({N_BLOWUP_STEPS} steps, nf={NF_BLOWUP})...')
    rA = run_TA_blowup(vk)
    for i, (v, gh) in enumerate(zip(rA['vals'][1:], rA['gnorms_hfn'])):
        flag = '^' if v > rA['val_before'] * 1.05 else ('v' if v < rA['val_before'] * 0.95 else '~')
        print(f'    step {i+1}: val={v:.5f} {flag}  hfn_grad={gh:.3e}  ({time.time()-t0:.0f}s)')

    t0 = time.time()
    print(f'  [TB] gradient cascade (nf={NF_CASCADE}, before any update)...')
    rB = run_TB_cascade(vk)
    print(f'    sqrt_loss={rB["sqrt_loss"]:.4e}')
    for name, gnorm, shape in rB['layer_gnorms']:
        print(f'      {name:30s}  {gnorm:.3e}  {shape}')
    print(f'    grad_phys={rB["grad_phys"]:.3e}  grad_aug={rB["grad_aug"]:.3e}')
    print(f'    hidden layers have zero grad: {rB["hidden_grads_zero"]}  ({time.time()-t0:.0f}s)')

    t0 = time.time()
    print(f'  [TC] ANN-only blowup (enc frozen, {N_BLOWUP_STEPS} steps, nf={NF_BLOWUP})...')
    rC = run_TC_ann_only_blowup(vk)
    for i, (v, gh) in enumerate(zip(rC['vals'][1:], rC['gnorms_hfn'])):
        flag = '^' if v > rC['vals'][0] * 1.05 else ('v' if v < rC['vals'][0] * 0.95 else '~')
        print(f'    step {i+1}: val={v:.5f} {flag}  hfn_grad={gh:.3e}  ({time.time()-t0:.0f}s)')

    t0 = time.time()
    print(f'  [TD] grad norm vs nf...')
    rD = run_TD_grad_vs_nf(vk)
    for nf in NF_SWEEP_TD:
        print(f'    nf={nf:4d}: hfn_grad={rD["gnorms_hfn"][nf]:.3e}  '
              f'val_after={rD["vals_after"][nf]:.5f}  ({time.time()-t0:.0f}s)')

    results[vk] = dict(TA=rA, TB=rB, TC=rC, TD=rD)

## ═══════════════════════════════════════════════════════════════════════════
## Plots
## ═══════════════════════════════════════════════════════════════════════════

COLORS = {'A_ffnn_tanh': 'C0', 'B_ffnn_identity': 'C1', 'C_linear': 'C2'}
SHORT_LABELS = {'A_ffnn_tanh': 'A: FFNN+Tanh', 'B_ffnn_identity': 'B: FFNN+Id', 'C_linear': 'C: Linear'}

# ── Plot TA: val sim-RMS blowup ─────────────────────────────────────────────
fig_TA, (ax_ta1, ax_ta2) = plt.subplots(1, 2, figsize=(12, 4))

for vk in VARIANT_DEFS:
    rA = results[vk]['TA']
    steps = np.arange(len(rA['vals']))
    ax_ta1.semilogy(steps, rA['vals'], 'o-', color=COLORS[vk], label=SHORT_LABELS[vk])
    ax_ta2.semilogy(np.arange(1, N_BLOWUP_STEPS + 1), rA['gnorms_hfn'],
                    's-', color=COLORS[vk], label=SHORT_LABELS[vk])

ax_ta1.set_xlabel('gradient step')
ax_ta1.set_ylabel('val sim-RMS (log scale)')
ax_ta1.set_title(f'TA: Val sim-RMS per step (nf={NF_BLOWUP})')
ax_ta1.legend(); ax_ta1.grid(True, which='both')

ax_ta2.set_xlabel('gradient step')
ax_ta2.set_ylabel('||grad ANN|| (log scale)')
ax_ta2.set_title(f'TA: ANN gradient norm per step (nf={NF_BLOWUP})')
ax_ta2.legend(); ax_ta2.grid(True, which='both')

fig_TA.suptitle('TA: Blowup test — 2 gradient steps at nf=400')
fig_TA.tight_layout()
fig_TA.savefig(os.path.join(SAVE_DIR, 'diag10_TA_blowup.png'), dpi=150)
plt.close('all')
print('\n  -> diag10_TA_blowup.png')

# ── Plot TB: gradient cascade (per-layer gnorms) ────────────────────────────
fig_TB, axes_TB = plt.subplots(1, 3, figsize=(15, 4))

for ax, vk in zip(axes_TB, VARIANT_DEFS):
    rB = results[vk]['TB']
    names  = [n for n, g, s in rB['layer_gnorms']]
    gnorms = [g for n, g, s in rB['layer_gnorms']]
    colors_bar = ['C3' if g == 0.0 else 'C2' for g in gnorms]
    x_pos = np.arange(len(names))
    ax.bar(x_pos, gnorms, color=colors_bar)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('||grad||')
    ax.set_title(f'{SHORT_LABELS[vk]}\ngrad_phys={rB["grad_phys"]:.2e}  '
                 f'grad_aug={rB["grad_aug"]:.2e}')
    ax.grid(True, axis='y')

fig_TB.suptitle(f'TB: Per-layer gradient norms at nf={NF_CASCADE} (before any weight update)\n'
                f'Red = zero grad (cascade dead), Green = non-zero')
fig_TB.tight_layout()
fig_TB.savefig(os.path.join(SAVE_DIR, 'diag10_TB_cascade.png'), dpi=150)
plt.close('all')
print('  -> diag10_TB_cascade.png')

# ── Plot TC: ANN-only blowup ────────────────────────────────────────────────
fig_TC, (ax_tc1, ax_tc2) = plt.subplots(1, 2, figsize=(12, 4))

for vk in VARIANT_DEFS:
    rC = results[vk]['TC']
    steps = np.arange(len(rC['vals']))
    ax_tc1.semilogy(steps, rC['vals'], 'o-', color=COLORS[vk], label=SHORT_LABELS[vk])
    ax_tc2.semilogy(np.arange(1, N_BLOWUP_STEPS + 1), rC['gnorms_hfn'],
                    's-', color=COLORS[vk], label=SHORT_LABELS[vk])

ax_tc1.set_xlabel('gradient step')
ax_tc1.set_ylabel('val sim-RMS (log scale)')
ax_tc1.set_title(f'TC: Val sim-RMS (encoder frozen, nf={NF_BLOWUP})')
ax_tc1.legend(); ax_tc1.grid(True, which='both')

ax_tc2.set_xlabel('gradient step')
ax_tc2.set_ylabel('||grad ANN|| (log scale)')
ax_tc2.set_title(f'TC: ANN gradient norm (encoder frozen, nf={NF_BLOWUP})')
ax_tc2.legend(); ax_tc2.grid(True, which='both')

fig_TC.suptitle('TC: ANN-only blowup (encoder frozen) — 2 gradient steps at nf=400')
fig_TC.tight_layout()
fig_TC.savefig(os.path.join(SAVE_DIR, 'diag10_TC_ann_only.png'), dpi=150)
plt.close('all')
print('  -> diag10_TC_ann_only.png')

# ── Plot TD: grad norm vs nf ────────────────────────────────────────────────
fig_TD, (ax_td1, ax_td2) = plt.subplots(1, 2, figsize=(12, 4))

for vk in VARIANT_DEFS:
    rD = results[vk]['TD']
    nfs = NF_SWEEP_TD
    ax_td1.loglog(nfs, [rD['gnorms_hfn'][n] for n in nfs],
                  'o-', color=COLORS[vk], label=SHORT_LABELS[vk])
    ax_td2.semilogx(nfs, [rD['vals_after'][n] for n in nfs],
                    'o-', color=COLORS[vk], label=SHORT_LABELS[vk])

ax_td1.set_xlabel('rollout length nf')
ax_td1.set_ylabel('||grad ANN|| (log scale)')
ax_td1.set_title('TD: ANN grad norm vs rollout length (1 step, log-log)')
ax_td1.legend(); ax_td1.grid(True, which='both')

ax_td2.set_xlabel('rollout length nf')
ax_td2.set_ylabel('val sim-RMS after 1 step')
ax_td2.set_title('TD: Val sim-RMS after 1 step vs rollout length')
ax_td2.legend(); ax_td2.grid(True)

fig_TD.suptitle('TD: Gradient norm vs rollout length — does blowup scale with nf?')
fig_TD.tight_layout()
fig_TD.savefig(os.path.join(SAVE_DIR, 'diag10_TD_grad_vs_nf.png'), dpi=150)
plt.close('all')
print('  -> diag10_TD_grad_vs_nf.png')

## ═══════════════════════════════════════════════════════════════════════════
## Summary
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('SUMMARY')
print('='*70)

# Header
hdr = f"{'Test':<12}" + ''.join(f'{SHORT_LABELS[vk]:>22}' for vk in VARIANT_DEFS)
print(hdr)
print('-' * len(hdr))

# TA: val before -> after
print(f"\n{'TA val before':<12}" +
      ''.join(f'{results[vk]["TA"]["val_before"]:>22.5f}' for vk in VARIANT_DEFS))
for step in range(1, N_BLOWUP_STEPS + 1):
    print(f"{'TA step'+str(step):<12}" +
          ''.join(f'{results[vk]["TA"]["vals"][step]:>22.5f}' for vk in VARIANT_DEFS))

# TA: hfn_grad per step
for step in range(N_BLOWUP_STEPS):
    print(f"{'TA hfn g'+str(step+1):<12}" +
          ''.join(f'{results[vk]["TA"]["gnorms_hfn"][step]:>22.3e}' for vk in VARIANT_DEFS))

# TB: hidden grads zero?
print(f"\n{'TB hid=0':<12}" +
      ''.join(f'{str(results[vk]["TB"]["hidden_grads_zero"]):>22}' for vk in VARIANT_DEFS))
print(f"{'TB grad_phys':<12}" +
      ''.join(f'{results[vk]["TB"]["grad_phys"]:>22.3e}' for vk in VARIANT_DEFS))
print(f"{'TB grad_aug':<12}" +
      ''.join(f'{results[vk]["TB"]["grad_aug"]:>22.3e}' for vk in VARIANT_DEFS))

# TC: val per step
print(f"\n{'TC val ep0':<12}" +
      ''.join(f'{results[vk]["TC"]["vals"][0]:>22.5f}' for vk in VARIANT_DEFS))
for step in range(1, N_BLOWUP_STEPS + 1):
    print(f"{'TC step'+str(step):<12}" +
          ''.join(f'{results[vk]["TC"]["vals"][step]:>22.5f}' for vk in VARIANT_DEFS))

# TD: hfn_grad at nf=400
print(f"\n{'TD nf=400 g':<12}" +
      ''.join(f'{results[vk]["TD"]["gnorms_hfn"][400]:>22.3e}' for vk in VARIANT_DEFS))
print(f"{'TD nf=400 v':<12}" +
      ''.join(f'{results[vk]["TD"]["vals_after"][400]:>22.5f}' for vk in VARIANT_DEFS))

print(f'\nAll results -> {SAVE_DIR}/')
print(f'[TIMING] Total: {time.time() - _t_script_start:.0f}s = '
      f'{(time.time() - _t_script_start) / 60:.1f}min')
