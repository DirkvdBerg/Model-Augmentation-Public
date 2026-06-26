"""
diag13_routing_isolation.py
---------------------------
Isolate which physical state rows cause blowup and whether gradient clipping
is a viable training-level fix.

Prerequisites (fixes vs diag12):
  - All 8 training trajectories (not just T1) -> proper ~0.002 initial RMS
  - baseline_states.npz encoder init (linear_map)

Tests (each: 1 gradient step at nf=400, blowup check + per-channel gradient):
  T_vel   ANN -> velocity rows [3,4,5] only
          Velocity channels had 10-15x largest gradients (diag9).
          Question: do they alone cause blowup?

  T_pos   ANN -> position rows [0,1,2] only
          Positions have smaller gradients and a direct path to y via Cd.
          Question: does smaller gradient + direct output path avoid blowup?

  T_clip  ANN -> all states [0..7] + clip_grad_norm_(max_norm=1.0)
          Question: is blowup a gradient-scale issue fixable by clipping,
          or does any update to physical state rows destabilise the dynamics?

Decision table:
  T_vel blows up AND T_pos stable  -> use T_pos routing (position rows)
  T_vel stable AND T_pos blows up  -> use T_vel routing (velocity rows)
  Both stable                      -> either routing viable; prefer T_pos
                                      (direct output path via Cd)
  Both blow up + T_clip stable     -> add clip_grad_norm_ to all-states routing
  All blow up                      -> architectural fix required (Option B/C)
"""

import os
import sys
import time
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

SEED     = 42
NX_PHYS  = 6     # q1, q2, q3, dq1, dq2, dq3
NX_ANN   = 2     # encoder always uses NX_ANN=2 augmented states
nu       = 3
ny       = 3
Y_OP     = None  # LPV self-scheduled

FS_ORIG  = 20000
FS_NEW   = 4000
D        = FS_ORIG // FS_NEW
TS_NEW   = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32

HP = dict(
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=2,
    nf=400,
    batch_size=256,
    lr=1e-4,
)

CLIP_MAX_NORM = 1.0  # for T_clip

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet',
                       'encoder-augmentation', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

## ═══════════════════════════════════════════════════════════════════════════
## Data — ALL 8 training trajectories
## ═══════════════════════════════════════════════════════════════════════════

TRAJ_DIR = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab', 'multisine')
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

def load_traj(f):
    d = loadmat(os.path.join(TRAJ_DIR, f), squeeze_me=True)
    return deepSI.System_data(
        u=_load_u(d)[::D].astype(DTYPE_NP),
        y=d['y'][::D].astype(DTYPE_NP),
        dt=TS_NEW,
    )

print(f'Loading data (all {len(TRAIN_FILES)} training trajectories)...')
train_list = [load_traj(f) for f in TRAIN_FILES]
train_data = deepSI.System_data_list(train_list)
val_data   = load_traj(VAL_FILE)
n_train    = sum(len(t.y) for t in train_list)
print(f'  train: {n_train} samples | val: {len(val_data.y)} samples')

## ═══════════════════════════════════════════════════════════════════════════
## Normalisation
## ═══════════════════════════════════════════════════════════════════════════

u_all   = np.concatenate([t.u for t in train_list])
y_all   = np.concatenate([t.y for t in train_list])
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

_STATE_LABELS = ['q1', 'q2', 'q3(Y)', 'dq1', 'dq2', 'dq3', 'aug0', 'aug1']

## ═══════════════════════════════════════════════════════════════════════════
## build_variant
## ═══════════════════════════════════════════════════════════════════════════

def build_variant(ann_ix):
    """Build SSE_Interconnect with ANN routed to state indices ann_ix.

    ann_ix : 1-D integer array, e.g. np.array([3,4,5]).
             ANN output dimension nw = len(ann_ix).
    Encoder always uses NX_ANN=2 augmented states (linear_map init).
    """
    nxd      = NX_PHYS + NX_ANN
    nw       = len(ann_ix)
    na       = 4 * NX_PHYS + 1   # Jan's rule: na=nb=4*nx+1
    nb       = na
    na_right = 1
    nb_right = 1

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
        nz=nxd + nu, nw=nw,
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

    fit_sys = SSE_Interconnect(
        interconnect=ic, na=na, nb=nb,
        na_right=na_right, nb_right=nb_right,
        e_net_kwargs={
            "n_nodes_per_layer": HP['n_nodes_per_layer'],
            "n_hidden_layers":   HP['n_hidden_layers'],
        },
    )
    fit_sys.norm.u0   = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0   = y0
    fit_sys.norm.ystd = ystd

    # Encoder init: linear_map from reconstructability (Hoekstra 2026)
    Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)
    baseline_npz = os.path.join(
        PROJECT_ROOT, 'data', 'gantry', 'baseline_simulations',
        'multisine_LPV', 'baseline_states.npz')
    if os.path.exists(baseline_npz):
        bl        = np.load(baseline_npz, allow_pickle=True)
        x_phys_all = np.concatenate(bl['x_train_phys'])
    else:
        x_phys_all = x_all
        print('  WARNING: baseline_states.npz not found, using finite-diff states')
    sd   = deepSI.System_data(u=u_all, y=y_all)
    sd.x = x_phys_all
    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        Ad, Bd, Cd_dt, Dd_dt, sd)

    fit_sys.encoder = linear_encoder_init_aug(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
        nx_aug=NX_ANN,
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        flag_linear_only=False,
        u_mean=u_mean, std_u=std_u,
        y0=y0, ystd=ystd, x_mean=x_mean, std_x=std_x,
    ).to(DTYPE_PT)

    fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
    fit_sys.hfn.to(DTYPE_PT)
    return fit_sys


def build_fresh(ann_ix):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    return build_variant(ann_ix)


def get_val_sim_rms(m):
    m.eval()
    return float(m.cal_validation_error(val_data, validation_measure='sim-RMS'))


## ═══════════════════════════════════════════════════════════════════════════
## run_test — blowup check (nf=400, 1 step) + per-channel gradient report
## ═══════════════════════════════════════════════════════════════════════════

def run_test(name, ann_ix, clip=False, clip_norm=CLIP_MAX_NORM):
    """Blowup check + per-channel gradient breakdown for one routing variant.

    Returns dict: val_before, val_after, ratio, ch_grads, grad_total.
    """
    nxd = NX_PHYS + NX_ANN
    print(f'\n{"="*60}')
    label = f'[0..{nxd-1}]+clip(norm={clip_norm})' if clip else str(list(ann_ix))
    print(f'{name}: ANN -> {label}')
    print(f'{"="*60}')
    t0 = time.time()

    rng = np.random.default_rng(SEED)
    m   = build_fresh(ann_ix)
    val_before = get_val_sim_rms(m)
    print(f'  val_before: {val_before:.5f}')

    data  = m.make_training_data(m.norm.transform(train_data), nf=HP['nf'])
    n     = len(data[0])
    idx   = rng.choice(n, HP['batch_size'], replace=False)
    batch = [torch.tensor(data[i][idx], dtype=DTYPE_PT) for i in range(len(data))]

    m.train()
    m.optimizer.zero_grad()
    loss = m.loss(*batch, nf=HP['nf'])
    loss.backward()

    # Per-channel gradients from ANN final layer
    ann_block   = next(b for b in m.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    final_layer = ann_block.net.net[-1]
    W_grad      = final_layer.weight.grad   # shape (nw, n_hidden)

    ch_grads   = {}
    grad_total = 0.0
    if W_grad is not None:
        row_norms = W_grad.norm(dim=1).detach().cpu().numpy()   # (nw,)
        for i, state_ix in enumerate(ann_ix):
            ch_grads[int(state_ix)] = float(row_norms[i])
        grad_total = float(W_grad.norm().item())

    print('  ANN final-layer gradient per output channel:')
    if ch_grads:
        for state_ix, g in sorted(ch_grads.items()):
            lbl = _STATE_LABELS[state_ix] if state_ix < len(_STATE_LABELS) else f'x[{state_ix}]'
            print(f'    x[{state_ix}] {lbl:8s}: {g:.3e}')
    else:
        print('    (no gradient — ANN disconnected from loss)')
    print(f'  grad_total: {grad_total:.3e}')

    if clip:
        all_params = list(m.encoder.parameters()) + list(m.hfn.parameters())
        gnorm_pre = sum(
            p.grad.detach().norm().item() ** 2
            for p in all_params if p.grad is not None
        ) ** 0.5
        clipped = torch.nn.utils.clip_grad_norm_(all_params, max_norm=clip_norm)
        print(f'  global grad_norm  before clip: {gnorm_pre:.3e}')
        print(f'  global grad_norm  after  clip: {float(clipped):.3e}  (max_norm={clip_norm})')

    m.optimizer.step()
    val_after = get_val_sim_rms(m)
    ratio     = val_after / val_before
    flag      = ('^ WORSE' if ratio > 1.05 else
                 ('v better' if ratio < 0.95 else '~ stable'))
    print(f'  val: {val_before:.5f} -> {val_after:.5f}  {flag}  (x{ratio:.2f})')

    verdict = 'PASS' if ratio < 1.05 else f'FAIL (x{ratio:.1f})'
    print(f'  [{verdict}]')
    print(f'  ({time.time()-t0:.0f}s)')

    return dict(val_before=val_before, val_after=val_after, ratio=ratio,
                ch_grads=ch_grads, grad_total=grad_total, verdict=verdict)


## ═══════════════════════════════════════════════════════════════════════════
## Prerequisite: verify initial val RMS with all 8 trajs
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('PREREQUISITE: Initial val RMS (all 8 trajectories, aug-only routing)')
print('='*60)
nxd      = NX_PHYS + NX_ANN
aug_only = np.arange(NX_PHYS, nxd)   # [6,7] — safe, no blowup
m_pre    = build_fresh(aug_only)
val_init = get_val_sim_rms(m_pre)
del m_pre
print(f'  Initial val sim-RMS: {val_init:.5f}')
if val_init < 0.01:
    print('  [OK] ~0.002 range — encoder init and all-8-traj loading correct.')
else:
    print(f'  [WARNING] {val_init:.4f} > 0.01 — encoder init may be incorrect.')
    print('  All routing comparisons below are unreliable if initial RMS is wrong.')

## ═══════════════════════════════════════════════════════════════════════════
## Tests
## ═══════════════════════════════════════════════════════════════════════════

results = {}

# T_vel: ANN -> velocity rows [3,4,5]
# Largest gradients in diag9 (2.4e4 dq1, 3.8e4 dq2) -- primary blowup suspect
results['T_vel'] = run_test('T_vel', np.array([3, 4, 5]), clip=False)

# T_pos: ANN -> position rows [0,1,2]
# Smaller gradients (~1.5e3); positions appear directly in y = Cd_norm @ x_phys
results['T_pos'] = run_test('T_pos', np.array([0, 1, 2]), clip=False)

# T_clip: ANN -> all 8 states + gradient clipping
# Same setup as the original blowup (diag9), but with clip_grad_norm_(1.0)
results['T_clip'] = run_test('T_clip', np.arange(nxd), clip=True,
                              clip_norm=CLIP_MAX_NORM)

## ═══════════════════════════════════════════════════════════════════════════
## Summary
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'  Initial val RMS (before any training): {val_init:.5f}')
print()
routing_labels = {
    'T_vel':  '[3,4,5] vel',
    'T_pos':  '[0,1,2] pos',
    'T_clip': f'[0..{nxd-1}] all+clip',
}
print(f'  {"Test":<10} {"Routing":<20} {"before":>8} {"after":>8} {"ratio":>6}  {"grad_total":>10}  result')
print(f'  {"-"*75}')
for name, r in results.items():
    print(f'  {name:<10} {routing_labels[name]:<20} '
          f'{r["val_before"]:>8.5f} {r["val_after"]:>8.5f} '
          f'{r["ratio"]:>6.2f}  {r["grad_total"]:>10.3e}  {r["verdict"]}')

print()
print('Per-channel gradient breakdown:')
for name, r in results.items():
    grads = r['ch_grads']
    if grads:
        parts = ', '.join(
            f'x[{ix}]={g:.2e}' for ix, g in sorted(grads.items()))
    else:
        parts = '(zero — disconnected)'
    print(f'  {name}: {parts}')

print()
print('Decision:')
vel_pass  = results['T_vel']['ratio']  < 1.05
pos_pass  = results['T_pos']['ratio']  < 1.05
clip_pass = results['T_clip']['ratio'] < 1.05

if vel_pass and pos_pass:
    print('  Both T_vel and T_pos stable.')
    print('  Recommended: T_pos routing — positions feed y directly via Cd.')
elif not vel_pass and pos_pass:
    print('  T_vel blows up; T_pos stable.')
    print('  -> Use T_pos routing (ANN -> position rows [0,1,2]).')
elif vel_pass and not pos_pass:
    print('  T_pos blows up; T_vel stable.')
    print('  -> Use T_vel routing (ANN -> velocity rows [3,4,5]).')
else:
    print('  Both T_vel and T_pos blow up.')
    if clip_pass:
        print('  T_clip stable -> gradient clipping (max_norm=1.0) prevents blowup.')
        print('  -> Apply clip_grad_norm_(max_norm=1.0) to all-states routing in '
              'gantry_interconnect_dynamic.py.')
    else:
        print('  T_clip also blows up -> clipping insufficient at max_norm=1.0.')
        print('  -> Architectural fix required:')
        print('       Option B: Gantry_State_Block reads x_aug as input to physics')
        print('       Option C: add x_aug directly to output equation (y += Caug @ x_aug)')

## ═══════════════════════════════════════════════════════════════════════════
## Save
## ═══════════════════════════════════════════════════════════════════════════

save_path = os.path.join(OUT_DIR, 'diag13_results.npz')
np.savez(save_path,
         val_init=np.array(val_init),
         **{f'{name}_before': np.array(r['val_before']) for name, r in results.items()},
         **{f'{name}_after':  np.array(r['val_after'])  for name, r in results.items()},
         **{f'{name}_ratio':  np.array(r['ratio'])      for name, r in results.items()},
         **{f'{name}_grad':   np.array(r['grad_total']) for name, r in results.items()},
         )
print(f'\nSaved: {save_path}')
print('Done.')
