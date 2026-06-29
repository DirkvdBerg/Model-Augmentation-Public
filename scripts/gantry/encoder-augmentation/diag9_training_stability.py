"""
diag9_training_stability.py
---------------------------
Diagnoses training instability in the augmented gantry model.

Hypothesis: val sim-RMS blows up from epoch 0 because nf=400 rollout
causes the unconstrained augmented state dynamics (x[6,7]) to diverge
during the first gradient step. A curriculum (short nf first) should
stabilise training.

Tests
-----
T1  nf sweep  (N_BATCHES steps per nf, independent fresh model each time)
    Train loss and gradient norms per batch for nf in [1, 5, 25, 100, 400].
    Pinpoints the critical rollout length where training explodes.

T2  ANN state blowup tracking  (nf=400, N_BATCHES steps)
    RMS(x[6]), RMS(x[7]) from a full-horizon simulation after each step.
    Val sim-RMS after each step.
    Confirms whether augmented states diverge in rollout.

T3  Jacobian spectral radius
    Max singular value of  J = d(ann_block(x,u)[6:8]) / d(x[6:8])
    at epoch 0 and after 1 batch step with nf=400 and with nf=1.
    Spectral radius > 1 => augmented feedback dynamics are locally unstable.

T4  Curriculum vs fixed nf
    Model A: nf schedule [1 -> 5 -> 25 -> 100 -> 400], N_BATCHES each.
    Model B: fixed nf=400 for 5*N_BATCHES total steps.
    Val sim-RMS recorded after each phase.

All tests reuse config / data / build_model from gantry_interconnect_dynamic.py.
Output goes to scripts/gantry/encoder-augmentation/diagnostics/.

Usage:
    conda run -n GraduationProject python \\
        scripts/gantry/encoder-augmentation/diag9_training_stability.py
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
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import SSE_Interconnect, Interconnect
from model_augmentation.fit_systems.blocks import (
    Gantry_State_Block, Linear_Output_Block, Static_ANN_Block,
)
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize

## ═══════════════════════════════════════════════════════════════════════════
## Diagnostic config
## ═══════════════════════════════════════════════════════════════════════════

N_BATCHES        = 2                           # gradient steps per test (T1-T4, T6, T11, T12)
N_BATCHES_SWEEP  = 1                           # steps for T9/T10 scale sweeps (directional only)
NF_SWEEP         = [1, 400]                    # T1
CURRICULUM_NF    = [1, 400]                    # T4: one phase per nf value
AUG_SCALES       = [0.01, 0.1, 1.0, 10.0, 100.0]   # T9, T10
NF_SWEEP_T14     = [2, 5, 25, 100, 400]        # T14

SAVE_DIR = os.path.join(SCRIPT_DIR, 'diagnostics')
os.makedirs(SAVE_DIR, exist_ok=True)

## ═══════════════════════════════════════════════════════════════════════════
## Config -identical to gantry_interconnect_dynamic.py
## ═══════════════════════════════════════════════════════════════════════════

MODE           = 'multisine'
NX_PHYS        = 6
nu             = 3
ny             = 3
Y_OP           = None
ENCODER_INIT   = 'linear_map'
ANN_ACTIVATION = 'tanh'
SEED           = 42

FS_ORIG = 20000
FS_NEW  = 4000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32

NF_SECONDS = 0.100

DEFAULT_HP = dict(
    NX_ANN=2,
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=2,
    nf=max(1, int(NF_SECONDS / TS_NEW)),   # 400
    na_nb=0,
    batch_size=256,
    lr=1e-4,
    epochs=10,
)
DEFAULT_HP['na_nb'] = (NX_PHYS + DEFAULT_HP['NX_ANN']) * 2 + 1

## ═══════════════════════════════════════════════════════════════════════════
## Data loading -identical to gantry_interconnect_dynamic.py
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
train_list = [load_traj(f) for f in TRAIN_FILES]
train_data = deepSI.System_data_list(train_list)
val_data   = load_traj(VAL_FILE)
_n_train = sum(len(t.y) for t in train_list)
_n_val   = len(val_data.y)
print(f'  {len(train_list)} train trajectories | val: {_n_val} samples')

## ═══════════════════════════════════════════════════════════════════════════
## Timing system
## ═══════════════════════════════════════════════════════════════════════════

import json as _json
from datetime import datetime as _datetime

_TIMING_DB = os.path.join(SAVE_DIR, 'diag9_timing.json')
_RUN_SCRIPT_START = time.time()
_run_timings = {}   # filled in as each test completes

_THIS_CONFIG = dict(
    N_BATCHES       = N_BATCHES,
    N_BATCHES_SWEEP = N_BATCHES_SWEEP,
    NF_SWEEP        = NF_SWEEP,
    CURRICULUM_NF   = CURRICULUM_NF,
    AUG_SCALES      = AUG_SCALES,
    NF_SWEEP_T14    = NF_SWEEP_T14,
    batch_size      = DEFAULT_HP['batch_size'],
    n_train         = _n_train,
    n_val           = _n_val,
)

def _work_units(cfg):
    """Comparable work counts per test for time scaling across configs."""
    nb  = cfg.get('N_BATCHES', 2)
    nbs = cfg.get('N_BATCHES_SWEEP', nb)
    return {
        'T1' : len(cfg.get('NF_SWEEP', [1,400])) * nb,
        'T2' : nb,
        'T3' : 2,
        'T4' : len(cfg.get('CURRICULUM_NF', [1,400])) * nb * 2,
        'T5' : 2,
        'T6' : nb,
        'T7' : 2,
        'T8' : 4,
        'T9' : len(cfg.get('AUG_SCALES', [1.0])) * nbs,
        'T10': len(cfg.get('AUG_SCALES', [1.0])) * nbs,
        'T11': nb,
        'T12': nb,
        'T13': 1,
        'T14': len(cfg.get('NF_SWEEP_T14', [400])),
    }

# Load timing DB and build estimates
_prev_timings = {}   # test -> estimated seconds
_prev_run_label = 'no history'
if os.path.exists(_TIMING_DB):
    try:
        with open(_TIMING_DB) as _f:
            _db = _json.load(_f)
        if _db.get('runs'):
            _prev = _db['runs'][-1]
            _prev_wus = _work_units(_prev['config'])
            _cur_wus  = _work_units(_THIS_CONFIG)
            for _t, _s in _prev['timings'].items():
                _pw = _prev_wus.get(_t, 1)
                _cw = _cur_wus.get(_t, _pw)
                _prev_timings[_t] = _s * (_cw / _pw) if _pw > 0 else _s
            _prev_run_label = _prev.get('timestamp', 'unknown')[:16]
    except Exception:
        pass

_est_total = sum(_prev_timings.values()) if _prev_timings else 0

def _timer_end(test_name, t_start):
    """Call at the end of each test block. Prints progress line."""
    elapsed = time.time() - t_start
    _run_timings[test_name] = round(elapsed, 1)
    total_so_far = time.time() - _RUN_SCRIPT_START
    remaining_est = max(0.0, _est_total - total_so_far)
    est_str = (f'remaining est: {remaining_est:.0f}s = {remaining_est/60:.1f}min'
               if _est_total > 0 else 'no estimate')
    print(f'[{test_name:3s} DONE] {elapsed:6.0f}s  |  '
          f'elapsed: {total_so_far:6.0f}s  |  {est_str}')

# Print header estimate
print(f'\n[TIMING] Based on: {_prev_run_label}')
if _prev_timings:
    _est_lines = ['  ' + '  '.join(
        f'{t}:{_prev_timings[t]:.0f}s' for t in list(_prev_timings.keys())[i:i+7]
    ) for i in range(0, len(_prev_timings), 7)]
    for _line in _est_lines:
        print(_line)
    print(f'  EST TOTAL: {_est_total:.0f}s = {_est_total/60:.1f} min')
else:
    print('  No previous timing data -- first run will record baseline.')

## ═══════════════════════════════════════════════════════════════════════════
## Normalisation -identical to gantry_interconnect_dynamic.py
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
## build_model -identical to gantry_interconnect_dynamic.py
## ═══════════════════════════════════════════════════════════════════════════

def _get_encoder_dims(hp):
    if ENCODER_INIT == 'linear_map':
        na = 4 * NX_PHYS + 1
        return na, na, 1, 1
    na = hp.get('na_nb', 2 * (NX_PHYS + hp['NX_ANN']) + 1)
    return na, na, 0, 0

def build_model(hp):
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

    _act = torch.nn.Identity if ANN_ACTIVATION == 'linear' else torch.nn.Tanh
    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=nxd,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=_act,
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

    if ENCODER_INIT == 'linear_map':
        Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)
        baseline_npz = os.path.join(
            PROJECT_ROOT, 'data', 'gantry', 'baseline_simulations',
            f'{MODE}_LPV', 'baseline_states.npz')
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

## ═══════════════════════════════════════════════════════════════════════════
## Diagnostic helpers
## ═══════════════════════════════════════════════════════════════════════════

def build_fresh():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    return build_model(DEFAULT_HP)

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
    gnorm_hfn = _gnorm(fit_sys.hfn)   # ANN only (physics is buffers)
    fit_sys.optimizer.step()
    return float(loss_val.item()) ** 0.5, gnorm_enc, gnorm_hfn

def get_val_sim_rms(fit_sys):
    fit_sys.eval()
    return float(fit_sys.cal_validation_error(val_data, validation_measure='sim-RMS'))

def get_ann_state_rms(fit_sys):
    """RMS of normalised x[6,7] from full simulation of val_data."""
    NX_ANN = DEFAULT_HP['NX_ANN']
    nxd    = NX_PHYS + NX_ANN
    fit_sys.eval()
    fit_sys.hfn.reset_saved_signals()
    with torch.no_grad():
        sim = fit_sys.apply_experiment(val_data)
    x_enc_norm = np.array(fit_sys.hfn.saved_output_signals)   # (nxd, T)
    x_ann      = x_enc_norm[NX_PHYS:nxd, :]                   # (NX_ANN, T)
    return np.sqrt((x_ann ** 2).mean(axis=1))                   # (NX_ANN,)

def jacobian_spectral_radius(fit_sys):
    """
    Max singular value of  J = d(ann_net output[NX_PHYS:]) / d(x_input[NX_PHYS:]).

    Uses a typical normalised (x, u) from the val encoder output.
    At epoch 0 J = 0 (zero-init final layer). After one step it becomes nonzero.
    Spectral radius > 1 means the x_aug feedback loop is locally unstable.
    """
    NX_ANN = DEFAULT_HP['NX_ANN']
    nxd    = NX_PHYS + NX_ANN
    na, nb, na_right, nb_right = _get_encoder_dims(DEFAULT_HP)

    ann_blk = next(
        b for b in fit_sys.hfn.connected_blocks if isinstance(b, Static_ANN_Block)
    )

    # Build a typical test point from the val encoder
    fit_sys.eval()
    val_norm = fit_sys.norm.transform(val_data)
    yn = np.ascontiguousarray(val_norm.y, dtype=DTYPE_NP)
    un = np.ascontiguousarray(val_norm.u, dtype=DTYPE_NP)
    k0 = na + 1
    yhist = torch.tensor(yn[k0 - na : k0 + na_right][None], dtype=DTYPE_PT)
    uhist = torch.tensor(un[k0 - nb : k0 + nb_right][None], dtype=DTYPE_PT)
    with torch.no_grad():
        x_enc = fit_sys.encoder(uhist, yhist)   # (1, nxd)
    u_t = torch.tensor(un[k0 : k0 + 1], dtype=DTYPE_PT)    # (1, nu)

    # Jacobian of ann_net(z)[NX_PHYS:] w.r.t. z[NX_PHYS:NX_PHYS+NX_ANN]
    x_phys_t  = x_enc[0, :NX_PHYS].detach()
    x_aug_var = x_enc[0, NX_PHYS:nxd].detach().clone().requires_grad_(True)
    u_flat    = u_t[0].detach()

    z_flat = torch.cat([x_phys_t, x_aug_var, u_flat]).unsqueeze(0)   # (1, nxd+nu)
    z_3d   = z_flat.unsqueeze(-1)                                      # (1, nxd+nu, 1)

    J_rows = []
    with torch.enable_grad():
        w = ann_blk(z_3d)                          # (1, nxd, 1)
        x_aug_out = w[0, NX_PHYS:nxd, 0]          # (NX_ANN,)
        for i in range(NX_ANN):
            g = torch.autograd.grad(
                x_aug_out[i], x_aug_var,
                retain_graph=True, create_graph=False,
            )[0]
            J_rows.append(g.detach().numpy().copy())

    J  = np.array(J_rows)          # (NX_ANN, NX_ANN)
    sv = np.linalg.svd(J, compute_uv=False)
    return float(sv[0]) if len(sv) > 0 else 0.0, J


## ═══════════════════════════════════════════════════════════════════════════
## T1: nf sweep
## ═══════════════════════════════════════════════════════════════════════════

_t_start = time.time()
print('\n' + '='*70)
print('T1: nf sweep')
print('='*70)

rng_t1    = np.random.default_rng(SEED)
t1_losses = {}      # nf -> list of sqrt-losses
t1_gnorm_enc = {}
t1_gnorm_hfn = {}
t1_val_before = {}
t1_val_after  = {}

for nf_val in NF_SWEEP:
    t0 = time.time()
    m  = build_fresh()
    t1_val_before[nf_val] = get_val_sim_rms(m)
    data = make_data(m, nf_val)

    losses, g_enc, g_hfn = [], [], []
    for _ in range(N_BATCHES):
        sl, ge, gh = manual_step(m, data, nf_val, rng_t1)
        losses.append(sl); g_enc.append(ge); g_hfn.append(gh)

    t1_val_after[nf_val]  = get_val_sim_rms(m)
    t1_losses[nf_val]     = losses
    t1_gnorm_enc[nf_val]  = g_enc
    t1_gnorm_hfn[nf_val]  = g_hfn

    direction = ('^ WORSE' if t1_val_after[nf_val] > t1_val_before[nf_val] * 1.05
                 else ('v better' if t1_val_after[nf_val] < t1_val_before[nf_val] * 0.95
                       else '~ stable'))
    print(f'  nf={nf_val:4d}: val {t1_val_before[nf_val]:.5f} -> {t1_val_after[nf_val]:.5f}  '
          f'{direction}  ({time.time()-t0:.0f}s)')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ax_l, ax_e, ax_h = axes
for nf_val in NF_SWEEP:
    lbl = f'nf={nf_val}'
    ax_l.semilogy(t1_losses[nf_val],    label=lbl)
    ax_e.semilogy(t1_gnorm_enc[nf_val], label=lbl)
    ax_h.semilogy(t1_gnorm_hfn[nf_val], label=lbl)
ax_l.set_title('sqrt(train loss) per batch')
ax_e.set_title('encoder grad norm per batch')
ax_h.set_title('ANN (hfn) grad norm per batch')
for ax in axes:
    ax.set_xlabel('batch'); ax.legend(fontsize=7); ax.grid(True, which='both')
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag9_T1_nf_sweep.png'), dpi=150)

fig2, ax2 = plt.subplots(figsize=(7, 3))
xs = np.arange(len(NF_SWEEP))
ax2.bar(xs - 0.2, [t1_val_before[n] for n in NF_SWEEP], 0.4, label='epoch 0', alpha=0.8)
ax2.bar(xs + 0.2, [t1_val_after[n]  for n in NF_SWEEP], 0.4, label=f'after {N_BATCHES} steps', alpha=0.8)
ax2.set_xticks(xs); ax2.set_xticklabels([f'nf={n}' for n in NF_SWEEP])
ax2.set_ylabel('val sim-RMS'); ax2.set_title(f'T1: Val sim-RMS before/after {N_BATCHES} steps')
ax2.legend(); ax2.grid(True, axis='y')
fig2.tight_layout()
fig2.savefig(os.path.join(SAVE_DIR, 'diag9_T1_val_rms.png'), dpi=150)
plt.close('all')
print('  -> diag9_T1_nf_sweep.png  diag9_T1_val_rms.png')
_timer_end('T1', _t_start)

## ═══════════════════════════════════════════════════════════════════════════
## T2: ANN state blowup (nf=400)
## ═══════════════════════════════════════════════════════════════════════════

_t_start = time.time()
print('\n' + '='*70)
print('T2: ANN state blowup per batch (nf=400)')
print('='*70)

rng_t2  = np.random.default_rng(SEED)
m2      = build_fresh()
data_t2 = make_data(m2, 400)
NX_ANN  = DEFAULT_HP['NX_ANN']

t2_ann_rms = []   # (N_BATCHES+1, NX_ANN)
t2_val_rms = []   # (N_BATCHES+1,)
t2_losses  = []   # (N_BATCHES,)

t2_ann_rms.append(get_ann_state_rms(m2).tolist())
t2_val_rms.append(get_val_sim_rms(m2))
print(f'  step  0: val={t2_val_rms[-1]:.5f}  ann_rms={[f"{v:.3e}" for v in t2_ann_rms[-1]]}')

for i in range(N_BATCHES):
    t0 = time.time()
    sl, ge, gh = manual_step(m2, data_t2, 400, rng_t2)
    t2_losses.append(sl)
    t2_ann_rms.append(get_ann_state_rms(m2).tolist())
    t2_val_rms.append(get_val_sim_rms(m2))
    print(f'  step {i+1:2d}: val={t2_val_rms[-1]:.5f}  ann_rms={[f"{v:.3e}" for v in t2_ann_rms[-1]]}'
          f'  sqrt_loss={sl:.5f}  grad enc={ge:.3e} hfn={gh:.3e}  ({time.time()-t0:.1f}s)')

ann_rms_arr = np.array(t2_ann_rms)  # (N_BATCHES+1, NX_ANN)

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4))
for ch in range(NX_ANN):
    ax3a.semilogy(ann_rms_arr[:, ch], marker='o', ms=4, label=f'x[{NX_PHYS+ch}]')
ax3a.set_xlabel('batch step (0 = epoch 0)')
ax3a.set_ylabel('RMS (normalised)')
ax3a.set_title('ANN state RMS in simulation (nf=400 training)')
ax3a.legend(); ax3a.grid(True, which='both')

ax3b.semilogy(t2_val_rms, marker='o', ms=4, color='C0')
ax3b.axhline(t2_val_rms[0], color='k', lw=0.8, ls='--',
             label=f'epoch-0  ({t2_val_rms[0]:.5f})')
ax3b.set_xlabel('batch step (0 = epoch 0)')
ax3b.set_ylabel('val sim-RMS')
ax3b.set_title('Val sim-RMS per batch (nf=400 training)')
ax3b.legend(); ax3b.grid(True, which='both')

fig3.suptitle(f'T2: ANN state blowup ({N_BATCHES} gradient steps, nf=400)')
fig3.tight_layout()
fig3.savefig(os.path.join(SAVE_DIR, 'diag9_T2_ann_blowup.png'), dpi=150)
plt.close('all')
print('  -> diag9_T2_ann_blowup.png')
_timer_end('T2', _t_start)

## ═══════════════════════════════════════════════════════════════════════════
## T3: Jacobian spectral radius  (epoch 0  vs  after 1 step nf=400/nf=1)
## ═══════════════════════════════════════════════════════════════════════════

_t_start = time.time()
print('\n' + '='*70)
print('T3: Jacobian spectral radius of x_aug feedback')
print('='*70)

rng_t3 = np.random.default_rng(SEED)

# ── epoch 0 ──────────────────────────────────────────────────────────────
m3a     = build_fresh()
sr0, J0 = jacobian_spectral_radius(m3a)
print(f'  Epoch 0 (all models): spectral radius = {sr0:.4e}')

# ── after 1 step, nf=400 ─────────────────────────────────────────────────
data_t3a = make_data(m3a, 400)
manual_step(m3a, data_t3a, 400, rng_t3)
sr_nf400, J_nf400 = jacobian_spectral_radius(m3a)
print(f'  After 1 step (nf=400): spectral radius = {sr_nf400:.4e}')
print(f'    J =\n{np.array2string(J_nf400, precision=4, suppress_small=True)}')

# ── after 1 step, nf=1 ───────────────────────────────────────────────────
m3b      = build_fresh()
data_t3b = make_data(m3b, 1)
manual_step(m3b, data_t3b, 1, rng_t3)
sr_nf1, J_nf1 = jacobian_spectral_radius(m3b)
print(f'  After 1 step  (nf=1):  spectral radius = {sr_nf1:.4e}')
print(f'    J =\n{np.array2string(J_nf1, precision=4, suppress_small=True)}')

t3_summary = dict(
    sr_epoch0=sr0,
    sr_after1_nf400=sr_nf400,
    sr_after1_nf1=sr_nf1,
    J_epoch0=J0.tolist(),
    J_nf400=J_nf400.tolist(),
    J_nf1=J_nf1.tolist(),
)
with open(os.path.join(SAVE_DIR, 'diag9_T3_jacobian.json'), 'w') as f:
    json.dump(t3_summary, f, indent=2)
print('  -> diag9_T3_jacobian.json')
_timer_end('T3', _t_start)

## ═══════════════════════════════════════════════════════════════════════════
## T4: Curriculum vs fixed nf=400
## ═══════════════════════════════════════════════════════════════════════════

_t_start = time.time()
print('\n' + '='*70)
print('T4: Curriculum vs fixed nf=400')
print('='*70)

rng_t4 = np.random.default_rng(SEED)

def run_schedule(phases, rng, tag):
    """Train with a sequence of (nf, n_steps) phases.
    Returns list of (label, val_sim_rms) tuples.
    """
    m = build_fresh()
    records = [('epoch0', float(get_val_sim_rms(m)))]
    print(f'  [{tag}] epoch0: val={records[-1][1]:.5f}')
    for nf_phase, n_steps in phases:
        data_phase = make_data(m, nf_phase)
        for _ in range(n_steps):
            manual_step(m, data_phase, nf_phase, rng)
        val = float(get_val_sim_rms(m))
        records.append((f'nf={nf_phase}', val))
        print(f'  [{tag}] after nf={nf_phase} ({n_steps} steps): val={val:.5f}')
    return records

curriculum_phases = [(nf, N_BATCHES) for nf in CURRICULUM_NF]
fixed_phases      = [(400, N_BATCHES * len(CURRICULUM_NF))]

curr_records  = run_schedule(curriculum_phases, rng_t4, 'curriculum')
fixed_records = run_schedule(fixed_phases,      rng_t4, 'fixed nf=400')

# Plot
fig4, ax4 = plt.subplots(figsize=(9, 4))
x_curr  = np.arange(len(curr_records))
vals_c  = [r[1] for r in curr_records]
lbls_c  = [r[0] for r in curr_records]
ax4.semilogy(x_curr, vals_c, marker='o', ms=5, label='Curriculum [1->5->25->100->400]')

# Map fixed_records onto same x-axis: epoch0 at 0, final at last curr index
x_fixed = [0, len(curr_records) - 1]
ax4.semilogy(x_fixed, [fixed_records[0][1], fixed_records[-1][1]],
             marker='s', ms=6, ls='--', label=f'Fixed nf=400 ({N_BATCHES*len(CURRICULUM_NF)} steps)')

ax4.set_xticks(x_curr); ax4.set_xticklabels(lbls_c, rotation=15, ha='right')
ax4.set_ylabel('val sim-RMS (log scale)')
ax4.set_title(f'T4: Curriculum vs fixed nf ({N_BATCHES} steps/phase)')
ax4.legend(); ax4.grid(True, which='both')
fig4.tight_layout()
fig4.savefig(os.path.join(SAVE_DIR, 'diag9_T4_curriculum.png'), dpi=150)
plt.close('all')
print('  -> diag9_T4_curriculum.png')
_timer_end('T4', _t_start)

## ═══════════════════════════════════════════════════════════════════════════
## T5: Gradient flow through ANN output rows (phys vs aug)
## ═══════════════════════════════════════════════════════════════════════════

_t_start = time.time()
print('\n' + '='*70)
print('T5: Gradient flow - ANN physical vs augmented output rows')
print('='*70)

rng_t5 = np.random.default_rng(SEED)

def gradient_flow_check(nf_val, rng):
    """Single backward pass. Returns per-layer gradient norms and the
    final-layer weight gradient split into physical rows [0:NX_PHYS] and
    augmented rows [NX_PHYS:nxd].

    Note: with nf=1 the ANN is NOT in the computation graph (yhat = C @ x_0
    depends only on the encoder; ANN only affects x_1 = xp, which is never
    used for y_pred in that single rollout step).  In that case all ANN
    gradients are None / 0 by design.  Use nf_grad >= 2 for the actual
    gradient-flow check.
    """
    NX_ANN_loc = DEFAULT_HP['NX_ANN']
    nxd_loc    = NX_PHYS + NX_ANN_loc

    # Use at least nf=2 so the ANN output (x_1) feeds into y_pred_2 and the
    # gradient can flow back to ANN weights.
    nf_grad = max(nf_val, 2)

    m    = build_fresh()
    data = make_data(m, nf_grad)

    n_total = len(data[0])
    idx   = rng.choice(n_total, DEFAULT_HP['batch_size'], replace=False)
    batch = [torch.tensor(d[idx], dtype=DTYPE_PT) for d in data]

    m.train()
    m.optimizer.zero_grad()
    loss = m.loss(*batch, nf=nf_grad)
    loss.backward()

    ann_blk = next(b for b in m.hfn.connected_blocks if isinstance(b, Static_ANN_Block))

    # Layer-by-layer gradient norms inside the ANN sequential net
    layer_gnorms = []
    for name, p in ann_blk.net.net.named_parameters():
        if p.grad is not None:
            layer_gnorms.append((name, p.grad.norm().item(), list(p.shape)))

    # Final output layer: split by output row
    final_layer = ann_blk.net.net[-1]
    W_grad = final_layer.weight.grad    # [nxd, n_nodes] or None if not in graph
    if W_grad is None:
        W_grad = torch.zeros(nxd_loc, final_layer.weight.shape[1])
    grad_phys = W_grad[:NX_PHYS, :].norm().item()
    grad_aug  = W_grad[NX_PHYS:nxd_loc, :].norm().item()
    ratio     = grad_phys / (grad_aug + 1e-12)

    # Per-output-row gradient norms (one value per output dimension)
    per_row = W_grad.norm(dim=1).detach().numpy()   # [nxd]

    # Encoder gradient norm
    gnorm_enc = sum(
        p.grad.detach().norm().item()**2
        for p in m.encoder.parameters() if p.grad is not None
    )**0.5

    sqrt_loss = float(loss.item())**0.5
    return dict(
        sqrt_loss   = sqrt_loss,
        layer_gnorms= layer_gnorms,
        grad_phys   = grad_phys,
        grad_aug    = grad_aug,
        ratio       = ratio,
        per_row     = per_row,
        gnorm_enc   = gnorm_enc,
        nf_used     = nf_grad,
    )

print(f'\n  nf=1 (grad computed at nf={max(1,2)} to ensure ANN in graph):')
t5_nf1   = gradient_flow_check(1,   rng_t5)
print(f'    sqrt_loss={t5_nf1["sqrt_loss"]:.4e}  enc_grad={t5_nf1["gnorm_enc"]:.3e}')
print(f'    ANN layer gradient norms:')
for name, gnorm, shape in t5_nf1['layer_gnorms']:
    print(f'      {name:30s}  {gnorm:.3e}  {shape}')
print(f'    Final layer: grad_phys={t5_nf1["grad_phys"]:.3e}  '
      f'grad_aug={t5_nf1["grad_aug"]:.3e}  ratio={t5_nf1["ratio"]:.1f}x')
print(f'    Per output row: {[f"{v:.2e}" for v in t5_nf1["per_row"]]}')

print('\n  nf=400:')
t5_nf400 = gradient_flow_check(400, rng_t5)
print(f'    sqrt_loss={t5_nf400["sqrt_loss"]:.4e}  enc_grad={t5_nf400["gnorm_enc"]:.3e}')
print(f'    ANN layer gradient norms:')
for name, gnorm, shape in t5_nf400['layer_gnorms']:
    print(f'      {name:30s}  {gnorm:.3e}  {shape}')
print(f'    Final layer: grad_phys={t5_nf400["grad_phys"]:.3e}  '
      f'grad_aug={t5_nf400["grad_aug"]:.3e}  ratio={t5_nf400["ratio"]:.1f}x')
print(f'    Per output row: {[f"{v:.2e}" for v in t5_nf400["per_row"]]}')

# Plot: per-output-row gradient norm for nf=1 vs nf=400
fig5, axes5 = plt.subplots(1, 2, figsize=(12, 4))
x_rows  = np.arange(NX_PHYS + DEFAULT_HP['NX_ANN'])
labels  = ['q1','q2','q3','dq1','dq2','dq3','delta_a','vdelta_a']

for ax, res, nf_val in zip(axes5, [t5_nf1, t5_nf400], [1, 400]):
    colors = ['C0' if i < NX_PHYS else 'C1' for i in x_rows]
    ax.bar(x_rows, res['per_row'], color=colors)
    ax.set_xticks(x_rows)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('||dL/dW_out[row,:]||')
    ax.set_title(f'nf={nf_val}: ANN output-row grad norms\n'
                 f'phys={res["grad_phys"]:.2e}  aug={res["grad_aug"]:.2e}  '
                 f'ratio={res["ratio"]:.0f}x')
    ax.grid(True, axis='y')

axes5[0].text(0.02, 0.98, 'Blue: physical [0:6]   Orange: augmented [6:8]',
              transform=axes5[0].transAxes, fontsize=7, va='top')
fig5.suptitle('T5: Where does gradient go in the ANN output layer?')
fig5.tight_layout()
fig5.savefig(os.path.join(SAVE_DIR, 'diag9_T5_gradient_flow.png'), dpi=150)
plt.close('all')
print('  -> diag9_T5_gradient_flow.png')
_timer_end('T5', _t_start)

## ═══════════════════════════════════════════════════════════════════════════
## T6: Masked ANN - does zeroing physical rows [0:NX_PHYS] fix the blowup?
## After 1 step at nf=400 (blowup), apply two forward hooks:
##   mask_phys: zero ann_out[:, 0:NX_PHYS]   -> only augmented rows active
##   mask_aug:  zero ann_out[:, NX_PHYS:]    -> only physical rows active (control)
## If masking physical rows restores val~0.002 -> physical rows are the cause.
## ═══════════════════════════════════════════════════════════════════════════

_t_start = time.time()
print('\n' + '='*70)
print('T6: Masked ANN (zero physical rows vs zero aug rows after 1 step nf=400)')
print('='*70)

rng_t6   = np.random.default_rng(SEED)
nxd_t6   = NX_PHYS + DEFAULT_HP['NX_ANN']

m6       = build_fresh()
ann_t6   = next(b for b in m6.hfn.connected_blocks if isinstance(b, Static_ANN_Block))

val_t6_epoch0 = get_val_sim_rms(m6)
print(f'  epoch 0:              val={val_t6_epoch0:.5f}')

data_t6 = make_data(m6, 400)
manual_step(m6, data_t6, 400, rng_t6)
val_t6_after = get_val_sim_rms(m6)
print(f'  after 1 step nf=400:  val={val_t6_after:.5f}  (blowup expected)')

def _hook_zero_phys(module, input, output):
    out = output.clone(); out[:, :NX_PHYS, :] = 0.0; return out

def _hook_zero_aug(module, input, output):
    out = output.clone(); out[:, NX_PHYS:nxd_t6, :] = 0.0; return out

h = ann_t6.register_forward_hook(_hook_zero_phys)
val_t6_mask_phys = get_val_sim_rms(m6)
h.remove()
print(f'  zero phys rows [0:6]: val={val_t6_mask_phys:.5f}  '
      f'-> physical rows are the culprit: '
      f'{"YES" if val_t6_mask_phys < val_t6_after * 0.5 else "NO"}')

h = ann_t6.register_forward_hook(_hook_zero_aug)
val_t6_mask_aug = get_val_sim_rms(m6)
h.remove()
print(f'  zero aug rows [6:8]:  val={val_t6_mask_aug:.5f}  '
      f'-> aug rows are the culprit: '
      f'{"YES" if val_t6_mask_aug < val_t6_after * 0.5 else "NO"}')

fig6, ax6 = plt.subplots(figsize=(7, 4))
labels_t6 = ['epoch0', 'after 1 step\nnf=400', 'zero phys\n[0:6]', 'zero aug\n[6:8]']
vals_t6   = [val_t6_epoch0, val_t6_after, val_t6_mask_phys, val_t6_mask_aug]
colors_t6 = ['C2', 'C3', 'C0', 'C1']
ax6.bar(labels_t6, vals_t6, color=colors_t6)
ax6.axhline(val_t6_epoch0, color='k', lw=1, ls='--', label=f'epoch0 baseline ({val_t6_epoch0:.4f})')
ax6.set_ylabel('val sim-RMS')
ax6.set_title('T6: Does masking ANN physical rows restore baseline performance?')
ax6.legend(fontsize=8); ax6.grid(True, axis='y')
fig6.tight_layout()
fig6.savefig(os.path.join(SAVE_DIR, 'diag9_T6_masked_ann.png'), dpi=150)
plt.close('all')
print('  -> diag9_T6_masked_ann.png')
_timer_end('T6', _t_start)

## ═══════════════════════════════════════════════════════════════════════════
## T7: FP subspace projection of ANN physical-row gradient
## Builds projection Pi onto output subspace of the frozen FP model via SVD
## of [A_bar | B_bar]. Decomposes ANN grad[0:NX_PHYS] into:
##   in-subspace  = Pi @ grad     (gradient duplicating FP dynamics)
##   orthogonal   = (I-Pi) @ grad (gradient in directions FP cannot cover)
## High frac_in means the ANN is being pushed to duplicate/corrupt the FP.
## ═══════════════════════════════════════════════════════════════════════════

_t_start = time.time()
print('\n' + '='*70)
print('T7: FP subspace projection of ANN physical-row gradient')
print('='*70)

rng_t7 = np.random.default_rng(SEED)

# Build FP subspace from normalized system matrices (same path as build_model)
_Ad7, _Bd7, _Cd7, _Dd7 = gantry_linearize_and_discretize(dt=TS_NEW)
_bl7 = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'baseline_simulations',
                    f'{MODE}_LPV', 'baseline_states.npz')
_x7  = (np.concatenate(np.load(_bl7, allow_pickle=True)['x_train_phys'])
        if os.path.exists(_bl7) else x_all)
_sd7 = deepSI.System_data(u=u_all, y=y_all); _sd7.x = _x7
_Ad_bar7, _Bd_bar7, _, _ = normalize_linear_ss_matrices(_Ad7, _Bd7, _Cd7, _Dd7, _sd7)
Ad_np7 = np.array(_Ad_bar7)
Bd_np7 = np.array(_Bd_bar7)

AB7    = np.hstack([Ad_np7, Bd_np7])        # (NX_PHYS, NX_PHYS+nu)
U7, S7, _ = np.linalg.svd(AB7, full_matrices=False)
rank7  = int(np.sum(S7 > 1e-6 * S7[0]))
Pi7    = U7[:, :rank7] @ U7[:, :rank7].T    # (NX_PHYS, NX_PHYS)
print(f'  FP subspace rank = {rank7}/{NX_PHYS}')
print(f'  Singular values: {np.array2string(S7, precision=3, suppress_small=True)}')

t7_results = {}
for nf_val in [1, 400]:
    nf_grad7 = max(nf_val, 2)   # need nf>=2 so ANN feeds into y_pred and gets a gradient
    m7   = build_fresh()
    d7   = make_data(m7, nf_grad7)
    n7   = len(d7[0])
    idx7 = rng_t7.choice(n7, DEFAULT_HP['batch_size'], replace=False)
    b7   = [torch.tensor(d7[i][idx7], dtype=DTYPE_PT) for i in range(len(d7))]

    m7.train(); m7.optimizer.zero_grad()
    loss7 = m7.loss(*b7, nf=nf_grad7)
    loss7.backward()

    ann7      = next(b for b in m7.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    W_grad7_raw = ann7.net.net[-1].weight.grad
    if W_grad7_raw is None:
        W_grad7_raw = torch.zeros(NX_PHYS + DEFAULT_HP['NX_ANN'], ann7.net.net[-1].weight.shape[1])
    W_grad7   = W_grad7_raw[:NX_PHYS, :].detach().numpy()  # (NX_PHYS, n_nodes)
    g_in7     = Pi7 @ W_grad7
    g_orth7   = (np.eye(NX_PHYS) - Pi7) @ W_grad7
    norm_tot7 = np.linalg.norm(W_grad7)
    norm_in7  = np.linalg.norm(g_in7)
    norm_orth7= np.linalg.norm(g_orth7)
    frac_in7  = norm_in7**2 / (norm_tot7**2 + 1e-12)

    t7_results[nf_val] = dict(
        norm_total=norm_tot7, norm_in=norm_in7,
        norm_orth=norm_orth7, frac_in=frac_in7,
    )
    print(f'  nf={nf_val:4d}: grad_phys={norm_tot7:.3e}  '
          f'in-subspace={norm_in7:.3e}  orthogonal={norm_orth7:.3e}  '
          f'frac_in={frac_in7:.1%}')

fig7, axes7 = plt.subplots(1, 2, figsize=(10, 4))
for ax7, nf_val in zip(axes7, [1, 400]):
    r7 = t7_results[nf_val]
    ax7.bar(['in FP\nsubspace', 'orthogonal\nto FP'], [r7['norm_in'], r7['norm_orth']],
            color=['C3', 'C2'])
    ax7.set_title(f'nf={nf_val}: ANN phys-row gradient decomposition\n'
                  f'{r7["frac_in"]:.0%} in FP subspace  (rank={rank7})')
    ax7.set_ylabel('||gradient component||')
    ax7.grid(True, axis='y')
fig7.suptitle('T7: How much of ANN physical-row gradient lies in FP output subspace?\n'
              '(high frac_in = ANN is being pushed to duplicate/corrupt FP physics)')
fig7.tight_layout()
fig7.savefig(os.path.join(SAVE_DIR, 'diag9_T7_fp_subspace.png'), dpi=150)
plt.close('all')
print('  -> diag9_T7_fp_subspace.png')
_timer_end('T7', _t_start)

## ═══════════════════════════════════════════════════════════════════════════
## T8: D-055 normalisation fix on vs off
## ═══════════════════════════════════════════════════════════════════════════
## Does disabling the u_off/y_off/x_off fix in linear_encoder_init_aug change
## how gradients flow through the physical vs augmented ANN rows?

_t_start = time.time()
print('\n' + '='*70)
print('T8: D-055 normalisation fix on vs off')
print('='*70)

def gradient_flow_nf(fit_sys, nf_val):
    """Single backward pass; returns grad_phys, grad_aug, ratio, per_row.
    Uses nf >= 2 so ANN output (x_1) feeds into y_pred and gradient flows back.
    """
    NX_ANN_t8 = DEFAULT_HP['NX_ANN']
    nxd_t8    = NX_PHYS + NX_ANN_t8
    nf_grad8  = max(nf_val, 2)
    rng8 = np.random.default_rng(SEED + 80)
    data8 = make_data(fit_sys, nf_grad8)
    n_total8 = len(data8[0])
    idx8     = rng8.choice(n_total8, DEFAULT_HP['batch_size'], replace=False)
    batch8   = [torch.tensor(data8[i][idx8], dtype=DTYPE_PT) for i in range(len(data8))]
    fit_sys.train()
    fit_sys.optimizer.zero_grad()
    loss8 = fit_sys.loss(*batch8, nf=nf_grad8)
    loss8.backward()
    ann8    = next(b for b in fit_sys.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    W_grad8 = ann8.net.net[-1].weight.grad          # (nxd, n_nodes) or None
    if W_grad8 is None:
        W_grad8 = torch.zeros(nxd_t8, ann8.net.net[-1].weight.shape[1])
    per_row8  = W_grad8.norm(dim=1).detach().numpy()
    grad_phys8 = float(W_grad8[:NX_PHYS, :].norm().item())
    grad_aug8  = float(W_grad8[NX_PHYS:nxd_t8, :].norm().item())
    ratio8     = grad_phys8 / (grad_aug8 + 1e-12)
    return grad_phys8, grad_aug8, ratio8, per_row8

t8_results = {}  # key: (fix_label, nf_val) -> dict

for fix_on in [True, False]:
    label8 = 'D055_on' if fix_on else 'D055_off'
    m8 = build_fresh()
    if not fix_on and hasattr(m8.encoder, 'fix_enabled'):
        m8.encoder.fix_enabled = False

    for nf_val in [1, 400]:
        gp, ga, ratio, pr = gradient_flow_nf(m8, nf_val)
        t8_results[(label8, nf_val)] = dict(
            grad_phys=gp, grad_aug=ga, ratio=ratio, per_row=pr,
        )
        print(f'  [{label8}] nf={nf_val:4d}: '
              f'grad_phys={gp:.3e}  grad_aug={ga:.3e}  ratio={ratio:.1f}x')

NX_ANN_t8 = DEFAULT_HP['NX_ANN']
nxd_t8    = NX_PHYS + NX_ANN_t8
row_labels8 = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3'] + [f'x_aug[{i}]' for i in range(NX_ANN_t8)]
colors8 = ['C0'] * NX_PHYS + ['C1'] * NX_ANN_t8

fig8, axes8 = plt.subplots(2, 2, figsize=(14, 8))
for row_i, fix_on in enumerate([True, False]):
    label8 = 'D055_on' if fix_on else 'D055_off'
    for col_i, nf_val in enumerate([1, 400]):
        ax8 = axes8[row_i, col_i]
        pr8 = t8_results[(label8, nf_val)]['per_row']
        ax8.bar(np.arange(nxd_t8), pr8, color=colors8)
        ax8.set_xticks(np.arange(nxd_t8))
        ax8.set_xticklabels(row_labels8, rotation=45, ha='right', fontsize=8)
        ax8.set_title(f'{label8}  nf={nf_val}  '
                      f'ratio={t8_results[(label8, nf_val)]["ratio"]:.1f}x')
        ax8.set_ylabel('||row grad||')
        ax8.grid(True, axis='y')
fig8.suptitle('T8: ANN final-layer gradient per output row\n'
              'Blue=physical [0:6]  Orange=augmented [6:8]')
fig8.tight_layout()
fig8.savefig(os.path.join(SAVE_DIR, 'diag9_T8_d055.png'), dpi=150)
plt.close('all')
print('  -> diag9_T8_d055.png')
_timer_end('T8', _t_start)

## ═══════════════════════════════════════════════════════════════════════════
## T9: std_aug scale sweep
## ═══════════════════════════════════════════════════════════════════════════
## A forward hook scales ANN output rows [NX_PHYS:] by k before they enter
## the interconnect. This simulates having a different normalisation for
## x[6:8] (the augmented states). We sweep k and check:
##   (a) val sim-RMS after N_BATCHES steps at nf=400
##   (b) grad_aug / grad_phys ratio (how well grad signal reaches ANN aug rows)

_t_start = time.time()
print('\n' + '='*70)
print('T9: std_aug scale sweep')
print('='*70)

NX_ANN_t9 = DEFAULT_HP['NX_ANN']
nxd_t9    = NX_PHYS + NX_ANN_t9

t9_val_before   = {}   # k -> val sim-RMS before training
t9_val_after    = {}   # k -> val sim-RMS after N_BATCHES steps
t9_ratio        = {}   # k -> grad_aug / grad_phys after last step
t9_per_row      = {}   # k -> per_row gradient norm after last step

for k9 in AUG_SCALES:
    m9 = build_fresh()

    def _hook_scale_aug(module, input, output, _k=k9):
        out9 = output.clone()
        out9[:, NX_PHYS:nxd_t9, :] = out9[:, NX_PHYS:nxd_t9, :] * _k
        return out9

    ann9     = next(b for b in m9.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    h9       = ann9.register_forward_hook(_hook_scale_aug)

    t9_val_before[k9] = get_val_sim_rms(m9)
    data_t9 = make_data(m9, 400)
    rng_t9  = np.random.default_rng(SEED + 90)

    last_gp9 = last_ga9 = last_pr9 = None
    for _step9 in range(N_BATCHES_SWEEP):
        n9     = len(data_t9[0])
        idx9   = rng_t9.choice(n9, DEFAULT_HP['batch_size'], replace=False)
        batch9 = [torch.tensor(data_t9[i][idx9], dtype=DTYPE_PT) for i in range(len(data_t9))]
        m9.train()
        m9.optimizer.zero_grad()
        loss9 = m9.loss(*batch9, nf=400)
        loss9.backward()
        W9   = ann9.net.net[-1].weight.grad          # (nxd, n_nodes) or None
        if W9 is None: W9 = torch.zeros(nxd_t9, ann9.net.net[-1].weight.shape[1])
        pr9  = W9.norm(dim=1).detach().numpy().copy()
        gp9  = float(W9[:NX_PHYS, :].norm().item())
        ga9  = float(W9[NX_PHYS:nxd_t9, :].norm().item())
        last_gp9, last_ga9, last_pr9 = gp9, ga9, pr9
        m9.optimizer.step()

    h9.remove()
    t9_val_after[k9]  = get_val_sim_rms(m9)
    t9_ratio[k9]      = last_ga9 / (last_gp9 + 1e-12)
    t9_per_row[k9]    = last_pr9

    flag9 = ('^' if t9_val_after[k9] > t9_val_before[k9] * 1.05
             else ('v' if t9_val_after[k9] < t9_val_before[k9] * 0.95 else '~'))
    print(f'  k={k9:7.2f}: val {t9_val_before[k9]:.5f} -> {t9_val_after[k9]:.5f}  {flag9}  '
          f'aug/phys ratio={t9_ratio[k9]:.3f}')

fig9, (ax9a, ax9b) = plt.subplots(1, 2, figsize=(12, 4))
ks9   = AUG_SCALES
vals9 = [t9_val_after[k] for k in ks9]
rat9  = [t9_ratio[k] for k in ks9]
ax9a.semilogx(ks9, vals9, 'o-', color='C0')
ax9a.axhline(list(t9_val_before.values())[0], ls='--', color='gray', label='epoch 0')
ax9a.set_xlabel('aug scale k')
ax9a.set_ylabel('val sim-RMS')
ax9a.set_title(f'T9: Val sim-RMS after {N_BATCHES_SWEEP} steps (nf=400)')
ax9a.legend(); ax9a.grid(True)
ax9b.semilogx(ks9, rat9, 's-', color='C1')
ax9b.axhline(1.0, ls='--', color='gray', label='ratio=1')
ax9b.set_xlabel('aug scale k')
ax9b.set_ylabel('grad_aug / grad_phys')
ax9b.set_title('T9: ANN augmented/physical gradient ratio vs scale')
ax9b.legend(); ax9b.grid(True)
fig9.tight_layout()
fig9.savefig(os.path.join(SAVE_DIR, 'diag9_T9_scale_sweep.png'), dpi=150)
plt.close('all')
print('  -> diag9_T9_scale_sweep.png')
_timer_end('T9', _t_start)

## ═══════════════════════════════════════════════════════════════════════════
## T10: Dynamic parallel + scale  (mask phys rows, scale aug rows)
## ═══════════════════════════════════════════════════════════════════════════
## Zero out ANN output rows [0:NX_PHYS] (ANN cannot correct physics states)
## and scale rows [NX_PHYS:] by k. This is the "dynamic parallel" structure
## (Hoekstra et al. 2025): ANN only drives augmented states.
## Sweep k to find whether a scale is needed to avoid aug-state invisibility.

_t_start = time.time()
print('\n' + '='*70)
print('T10: dynamic parallel (mask phys, scale aug) sweep')
print('='*70)

NX_ANN_t10 = DEFAULT_HP['NX_ANN']
nxd_t10    = NX_PHYS + NX_ANN_t10

t10_val_before  = {}
t10_val_after   = {}
t10_ratio       = {}

for k10 in AUG_SCALES:
    m10 = build_fresh()

    def _hook_parallel(module, input, output, _k=k10):
        out10 = output.clone()
        out10[:, :NX_PHYS, :]          = 0.0
        out10[:, NX_PHYS:nxd_t10, :] = out10[:, NX_PHYS:nxd_t10, :] * _k
        return out10

    ann10   = next(b for b in m10.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    h10     = ann10.register_forward_hook(_hook_parallel)

    t10_val_before[k10] = get_val_sim_rms(m10)
    data_t10 = make_data(m10, 400)
    rng_t10  = np.random.default_rng(SEED + 100)

    last_gp10 = last_ga10 = None
    for _step10 in range(N_BATCHES_SWEEP):
        n10     = len(data_t10[0])
        idx10   = rng_t10.choice(n10, DEFAULT_HP['batch_size'], replace=False)
        batch10 = [torch.tensor(data_t10[i][idx10], dtype=DTYPE_PT) for i in range(len(data_t10))]
        m10.train()
        m10.optimizer.zero_grad()
        loss10 = m10.loss(*batch10, nf=400)
        loss10.backward()
        W10  = ann10.net.net[-1].weight.grad         # (nxd, n_nodes) or None
        if W10 is None: W10 = torch.zeros(nxd_t10, ann10.net.net[-1].weight.shape[1])
        gp10 = float(W10[:NX_PHYS, :].norm().item())
        ga10 = float(W10[NX_PHYS:nxd_t10, :].norm().item())
        last_gp10, last_ga10 = gp10, ga10
        m10.optimizer.step()

    h10.remove()
    t10_val_after[k10] = get_val_sim_rms(m10)
    t10_ratio[k10]     = last_ga10 / (last_gp10 + 1e-12)

    flag10 = ('^' if t10_val_after[k10] > t10_val_before[k10] * 1.05
              else ('v' if t10_val_after[k10] < t10_val_before[k10] * 0.95 else '~'))
    print(f'  k={k10:7.2f}: val {t10_val_before[k10]:.5f} -> {t10_val_after[k10]:.5f}  {flag10}  '
          f'aug/phys ratio={t10_ratio[k10]:.3f}')

fig10, (ax10a, ax10b) = plt.subplots(1, 2, figsize=(12, 4))
ks10   = AUG_SCALES
vals10 = [t10_val_after[k] for k in ks10]
rat10  = [t10_ratio[k] for k in ks10]
ax10a.semilogx(ks10, vals10, 'o-', color='C0')
ax10a.axhline(list(t10_val_before.values())[0], ls='--', color='gray', label='epoch 0')
ax10a.set_xlabel('aug scale k')
ax10a.set_ylabel('val sim-RMS')
ax10a.set_title(f'T10: Dynamic parallel val sim-RMS after {N_BATCHES_SWEEP} steps')
ax10a.legend(); ax10a.grid(True)
ax10b.semilogx(ks10, rat10, 's-', color='C1')
ax10b.axhline(1.0, ls='--', color='gray', label='ratio=1')
ax10b.set_xlabel('aug scale k')
ax10b.set_ylabel('grad_aug / grad_phys')
ax10b.set_title('T10: ANN aug/phys gradient ratio (dynamic parallel)')
ax10b.legend(); ax10b.grid(True)
fig10.tight_layout()
fig10.savefig(os.path.join(SAVE_DIR, 'diag9_T10_dynamic_parallel.png'), dpi=150)
plt.close('all')
print('  -> diag9_T10_dynamic_parallel.png')
_timer_end('T10', _t_start)

## ═══════════════════════════════════════════════════════════════════════════
## T11: Freeze ANN weights, train encoder only
## ═══════════════════════════════════════════════════════════════════════════
## If val still blows up with ANN frozen -> encoder update alone is culprit.
## If val stays stable -> ANN update is necessary for the blowup.

_t_start = time.time()
print('\n' + '='*70)
print('T11: Freeze ANN, train encoder only (nf=400)')
print('='*70)

m11 = build_fresh()
ann11 = next(b for b in m11.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
for p in ann11.parameters():
    p.requires_grad_(False)

data_t11 = make_data(m11, 400)
rng_t11  = np.random.default_rng(SEED + 110)
t11_val      = [get_val_sim_rms(m11)]
t11_gnorm_enc = []
t11_gnorm_hfn = []

for step11 in range(N_BATCHES):
    sl11, ge11, gh11 = manual_step(m11, data_t11, 400, rng_t11)
    t11_val.append(get_val_sim_rms(m11))
    t11_gnorm_enc.append(ge11)
    t11_gnorm_hfn.append(gh11)
    flag11 = ('^' if t11_val[-1] > t11_val[0] * 1.05
              else ('v' if t11_val[-1] < t11_val[0] * 0.95 else '~'))
    print(f'  step {step11+1}: val={t11_val[-1]:.5f} {flag11}  '
          f'enc_grad={ge11:.3e}  hfn_grad={gh11:.3e}')

fig11, ax11 = plt.subplots(figsize=(7, 3))
ax11.plot(t11_val, 'o-', label='enc only (ANN frozen)')
ax11.axhline(t11_val[0], ls='--', color='gray', label='epoch 0')
ax11.set_xlabel('gradient step')
ax11.set_ylabel('val sim-RMS')
ax11.set_title('T11: Freeze ANN - encoder-only update at nf=400')
ax11.legend(); ax11.grid(True)
fig11.tight_layout()
fig11.savefig(os.path.join(SAVE_DIR, 'diag9_T11_freeze_ann.png'), dpi=150)
plt.close('all')
print('  -> diag9_T11_freeze_ann.png')
_timer_end('T11', _t_start)

## ═══════════════════════════════════════════════════════════════════════════
## T12: Freeze encoder weights, train ANN only
## ═══════════════════════════════════════════════════════════════════════════
## If val still blows up with encoder frozen -> ANN update alone is culprit.

_t_start = time.time()
print('\n' + '='*70)
print('T12: Freeze encoder, train ANN only (nf=400)')
print('='*70)

m12 = build_fresh()
for p in m12.encoder.parameters():
    p.requires_grad_(False)

data_t12 = make_data(m12, 400)
rng_t12  = np.random.default_rng(SEED + 120)
t12_val      = [get_val_sim_rms(m12)]
t12_gnorm_enc = []
t12_gnorm_hfn = []

for step12 in range(N_BATCHES):
    sl12, ge12, gh12 = manual_step(m12, data_t12, 400, rng_t12)
    t12_val.append(get_val_sim_rms(m12))
    t12_gnorm_enc.append(ge12)
    t12_gnorm_hfn.append(gh12)
    flag12 = ('^' if t12_val[-1] > t12_val[0] * 1.05
              else ('v' if t12_val[-1] < t12_val[0] * 0.95 else '~'))
    print(f'  step {step12+1}: val={t12_val[-1]:.5f} {flag12}  '
          f'enc_grad={ge12:.3e}  hfn_grad={gh12:.3e}')

fig12, ax12 = plt.subplots(figsize=(7, 3))
ax12.plot(t12_val, 's-', label='ann only (encoder frozen)')
ax12.axhline(t12_val[0], ls='--', color='gray', label='epoch 0')
ax12.set_xlabel('gradient step')
ax12.set_ylabel('val sim-RMS')
ax12.set_title('T12: Freeze encoder - ANN-only update at nf=400')
ax12.legend(); ax12.grid(True)
fig12.tight_layout()
fig12.savefig(os.path.join(SAVE_DIR, 'diag9_T12_freeze_enc.png'), dpi=150)
plt.close('all')
print('  -> diag9_T12_freeze_enc.png')
_timer_end('T12', _t_start)

## ═══════════════════════════════════════════════════════════════════════════
## T13: State norm per rollout step (forward pass only, no gradient step)
## ═══════════════════════════════════════════════════════════════════════════
## Tracks ||x_t|| for t=0..400 on a fresh model to check whether the physics
## dynamics themselves are divergent before any gradient update happens.
## Exponential growth -> forward-pass instability independent of training.

_t_start = time.time()
print('\n' + '='*70)
print('T13: State norm per rollout step (epoch 0, no gradient)')
print('='*70)

_na13, _nb13, _na_right13, _nb_right13 = _get_encoder_dims(DEFAULT_HP)
NX_ANN_t13 = DEFAULT_HP['NX_ANN']
nxd_t13    = NX_PHYS + NX_ANN_t13

m13 = build_fresh()
m13.eval()

_val_norm13 = m13.norm.transform(val_data)
_yn13 = np.ascontiguousarray(_val_norm13.y, dtype=DTYPE_NP)
_un13 = np.ascontiguousarray(_val_norm13.u, dtype=DTYPE_NP)
_k0_13 = _na13 + 1

with torch.no_grad():
    _yhist13 = torch.tensor(_yn13[_k0_13-_na13 : _k0_13+_na_right13][None], dtype=DTYPE_PT)
    _uhist13 = torch.tensor(_un13[_k0_13-_nb13 : _k0_13+_nb_right13][None], dtype=DTYPE_PT)
    _x13 = m13.encoder(_uhist13, _yhist13)   # (1, nxd)

    _T13 = min(400, len(_un13) - _k0_13 - 1)
    t13_x_norms    = [float(_x13.norm().item())]
    t13_phys_norms = [float(_x13[0, :NX_PHYS].norm().item())]
    t13_aug_norms  = [float(_x13[0, NX_PHYS:].norm().item())]

    for _t13 in range(_T13):
        _u_t13 = torch.tensor(_un13[_k0_13+_t13:_k0_13+_t13+1], dtype=DTYPE_PT)
        _, _x13 = m13.hfn(_x13, _u_t13)
        t13_x_norms.append(float(_x13.norm().item()))
        t13_phys_norms.append(float(_x13[0, :NX_PHYS].norm().item()))
        t13_aug_norms.append(float(_x13[0, NX_PHYS:].norm().item()))

print(f'  Steps tracked: {_T13}')
print(f'  ||x||:    t=0: {t13_x_norms[0]:.3e}  t={_T13}: {t13_x_norms[-1]:.3e}  '
      f'max: {max(t13_x_norms):.3e}')
print(f'  ||x_phys||: t=0: {t13_phys_norms[0]:.3e}  t={_T13}: {t13_phys_norms[-1]:.3e}')
print(f'  ||x_aug||:  t=0: {t13_aug_norms[0]:.3e}  t={_T13}: {t13_aug_norms[-1]:.3e}')

fig13, ax13 = plt.subplots(figsize=(10, 3))
ts13 = np.arange(_T13 + 1)
ax13.plot(ts13, t13_x_norms,    label='||x|| total')
ax13.plot(ts13, t13_phys_norms, label='||x_phys|| [0:6]')
ax13.plot(ts13, t13_aug_norms,  label='||x_aug|| [6:8]')
ax13.set_xlabel('rollout step t')
ax13.set_ylabel('state norm')
ax13.set_title('T13: State norm per rollout step (epoch 0, no grad update)\n'
               'Exponential growth = forward-pass instability')
ax13.legend(); ax13.grid(True)
fig13.tight_layout()
fig13.savefig(os.path.join(SAVE_DIR, 'diag9_T13_state_norms.png'), dpi=150)
plt.close('all')
print('  -> diag9_T13_state_norms.png')
_timer_end('T13', _t_start)

## ═══════════════════════════════════════════════════════════════════════════
## T14: Gradient norm vs rollout length (nf sweep, single step each)
## ═══════════════════════════════════════════════════════════════════════════
## Builds a fresh model for each nf value, takes one gradient step, and
## records gnorm_enc and gnorm_hfn. Shows how gradient magnitude scales with
## rollout length -- superlinear growth points to BPTT gradient explosion.

_t_start = time.time()
print('\n' + '='*70)
print('T14: Gradient norm vs rollout length (1 step each)')
print('='*70)

t14_gnorm_enc  = {}
t14_gnorm_hfn  = {}
t14_val_before = {}
t14_val_after  = {}

for nf14 in NF_SWEEP_T14:
    m14   = build_fresh()
    t14_val_before[nf14] = get_val_sim_rms(m14)
    data14 = make_data(m14, nf14)
    rng14  = np.random.default_rng(SEED + 140)
    sl14, ge14, gh14 = manual_step(m14, data14, nf14, rng14)
    t14_val_after[nf14]  = get_val_sim_rms(m14)
    t14_gnorm_enc[nf14]  = ge14
    t14_gnorm_hfn[nf14]  = gh14
    flag14 = ('^' if t14_val_after[nf14] > t14_val_before[nf14] * 1.05
              else ('v' if t14_val_after[nf14] < t14_val_before[nf14] * 0.95 else '~'))
    print(f'  nf={nf14:4d}: enc_grad={ge14:.3e}  hfn_grad={gh14:.3e}  '
          f'val {t14_val_before[nf14]:.5f} -> {t14_val_after[nf14]:.5f}  {flag14}')

fig14, (ax14a, ax14b) = plt.subplots(1, 2, figsize=(12, 4))
nfs14 = NF_SWEEP_T14
ax14a.loglog(nfs14, [t14_gnorm_enc[n] for n in nfs14], 'o-', label='encoder')
ax14a.loglog(nfs14, [t14_gnorm_hfn[n] for n in nfs14], 's-', label='ANN (hfn)')
ax14a.set_xlabel('rollout length nf')
ax14a.set_ylabel('gradient norm')
ax14a.set_title('T14: Gradient norm vs rollout length (1 step, log-log)')
ax14a.legend(); ax14a.grid(True, which='both')
ax14b.semilogx(nfs14, [t14_val_after[n] for n in nfs14], 'o-', color='C0')
ax14b.axhline(list(t14_val_before.values())[0], ls='--', color='gray', label='epoch 0')
ax14b.set_xlabel('rollout length nf')
ax14b.set_ylabel('val sim-RMS after 1 step')
ax14b.set_title('T14: Val sim-RMS after 1 step vs rollout length')
ax14b.legend(); ax14b.grid(True)
fig14.tight_layout()
fig14.savefig(os.path.join(SAVE_DIR, 'diag9_T14_nf_grad_scaling.png'), dpi=150)
plt.close('all')
print('  -> diag9_T14_nf_grad_scaling.png')
_timer_end('T14', _t_start)

## ═══════════════════════════════════════════════════════════════════════════
## Summary
## ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('SUMMARY')
print('='*70)

print(f'\nT1 -val sim-RMS after {N_BATCHES} steps:')
for nf_val in NF_SWEEP:
    before = t1_val_before[nf_val]
    after  = t1_val_after[nf_val]
    ratio  = after / before
    flag   = '^ WORSE' if ratio > 1.05 else ('v better' if ratio < 0.95 else '~ stable')
    print(f'  nf={nf_val:4d}: {before:.5f} -> {after:.5f}  (x{ratio:.2f})  {flag}')

print(f'\nT2 -ANN state RMS after {N_BATCHES} steps (nf=400):')
for ch in range(NX_ANN):
    print(f'  x[{NX_PHYS+ch}]: {ann_rms_arr[0, ch]:.3e} -> {ann_rms_arr[-1, ch]:.3e}')
print(f'  val sim-RMS: {t2_val_rms[0]:.5f} -> {t2_val_rms[-1]:.5f}')

print(f'\nT3 - Jacobian spectral radius (x_aug -> x_aug):')
print(f'  epoch 0:          {sr0:.4e}  (expected 0 - zero-init final layer)')
print(f'  after 1 step nf=400:  {sr_nf400:.4e}')
print(f'  after 1 step nf=1:    {sr_nf1:.4e}')

print(f'\nT4 -val sim-RMS comparison:')
print(f'  curriculum final: {curr_records[-1][1]:.5f}')
print(f'  fixed nf=400:     {fixed_records[-1][1]:.5f}')
print(f'  {"Curriculum wins" if curr_records[-1][1] < fixed_records[-1][1] else "Fixed wins / no difference"}')

print(f'\nT5 - gradient flow (single backward pass):')
for tag, res, nf_val in [('nf=1  ', t5_nf1, 1), ('nf=400', t5_nf400, 400)]:
    print(f'  {tag}  grad_phys={res["grad_phys"]:.3e}  grad_aug={res["grad_aug"]:.3e}  '
          f'ratio={res["ratio"]:.1f}x  enc={res["gnorm_enc"]:.3e}')

print(f'\nT6 - masked ANN test (after 1 step nf=400):')
print(f'  epoch0={val_t6_epoch0:.5f}  after_step={val_t6_after:.5f}  '
      f'zero_phys={val_t6_mask_phys:.5f}  zero_aug={val_t6_mask_aug:.5f}')
phys_culprit = val_t6_mask_phys < val_t6_after * 0.5
print(f'  Physical rows [0:6] are the culprit: {"YES" if phys_culprit else "NO"}')

print(f'\nT7 - FP subspace projection (rank={rank7}):')
for nf_val in [1, 400]:
    r = t7_results[nf_val]
    print(f'  nf={nf_val:4d}: frac_in_FP_subspace={r["frac_in"]:.1%}  '
          f'grad_total={r["norm_total"]:.3e}')

print(f'\nT8 - D-055 on vs off (single backward pass):')
for fix_label in ['D055_on', 'D055_off']:
    for nf_val in [1, 400]:
        r8 = t8_results[(fix_label, nf_val)]
        print(f'  [{fix_label}] nf={nf_val:4d}: '
              f'grad_phys={r8["grad_phys"]:.3e}  grad_aug={r8["grad_aug"]:.3e}  '
              f'ratio={r8["ratio"]:.1f}x')

print(f'\nT9 - std_aug scale sweep (nf=400, {N_BATCHES_SWEEP} steps):')
for k9 in AUG_SCALES:
    print(f'  k={k9:7.2f}: val {t9_val_before[k9]:.5f} -> {t9_val_after[k9]:.5f}  '
          f'aug/phys={t9_ratio[k9]:.3f}')

print(f'\nT10 - dynamic parallel (mask phys + scale aug, nf=400, {N_BATCHES_SWEEP} steps):')
for k10 in AUG_SCALES:
    print(f'  k={k10:7.2f}: val {t10_val_before[k10]:.5f} -> {t10_val_after[k10]:.5f}  '
          f'aug/phys={t10_ratio[k10]:.3f}')

print(f'\nT11 - freeze ANN, encoder-only update (nf=400):')
for i, v in enumerate(t11_val):
    flag = ('epoch0' if i == 0 else f'step {i}')
    print(f'  {flag}: val={v:.5f}')

print(f'\nT12 - freeze encoder, ANN-only update (nf=400):')
for i, v in enumerate(t12_val):
    flag = ('epoch0' if i == 0 else f'step {i}')
    print(f'  {flag}: val={v:.5f}')

print(f'\nT13 - state norms over rollout (epoch 0):')
print(f'  ||x_phys||: t=0: {t13_phys_norms[0]:.3e}  t={_T13}: {t13_phys_norms[-1]:.3e}  '
      f'max: {max(t13_phys_norms):.3e}')
print(f'  ||x_aug||:  t=0: {t13_aug_norms[0]:.3e}  t={_T13}: {t13_aug_norms[-1]:.3e}')

print(f'\nT14 - gradient norm vs rollout length (1 step each):')
for nf14 in NF_SWEEP_T14:
    print(f'  nf={nf14:4d}: enc_grad={t14_gnorm_enc[nf14]:.3e}  '
          f'hfn_grad={t14_gnorm_hfn[nf14]:.3e}  '
          f'val {t14_val_before[nf14]:.5f} -> {t14_val_after[nf14]:.5f}')

# Save all numeric results
np.savez(
    os.path.join(SAVE_DIR, 'diag9_results.npz'),
    t1_nf_sweep               = np.array(NF_SWEEP),
    t1_val_before             = np.array([t1_val_before[n] for n in NF_SWEEP]),
    t1_val_after              = np.array([t1_val_after[n]  for n in NF_SWEEP]),
    t1_losses                 = np.array([t1_losses[n]     for n in NF_SWEEP]),
    t1_gnorm_enc              = np.array([t1_gnorm_enc[n]  for n in NF_SWEEP]),
    t1_gnorm_hfn              = np.array([t1_gnorm_hfn[n]  for n in NF_SWEEP]),
    t2_ann_rms                = ann_rms_arr,
    t2_val_rms                = np.array(t2_val_rms),
    t2_losses                 = np.array(t2_losses),
    t3_sr_epoch0              = np.array(sr0),
    t3_sr_nf400               = np.array(sr_nf400),
    t3_sr_nf1                 = np.array(sr_nf1),
    t4_curriculum_vals        = np.array([r[1] for r in curr_records]),
    t4_fixed_vals             = np.array([r[1] for r in fixed_records]),
    t5_per_row_nf1            = t5_nf1['per_row'],
    t5_per_row_nf400          = t5_nf400['per_row'],
    t5_grad_phys_nf1          = np.array(t5_nf1['grad_phys']),
    t5_grad_aug_nf1           = np.array(t5_nf1['grad_aug']),
    t5_grad_phys_nf400        = np.array(t5_nf400['grad_phys']),
    t5_grad_aug_nf400         = np.array(t5_nf400['grad_aug']),
    t6_val_epoch0             = np.array(val_t6_epoch0),
    t6_val_after_step         = np.array(val_t6_after),
    t6_val_mask_phys          = np.array(val_t6_mask_phys),
    t6_val_mask_aug           = np.array(val_t6_mask_aug),
    t7_rank                   = np.array(rank7),
    t7_singular_values        = S7,
    t7_frac_in_nf1            = np.array(t7_results[1]['frac_in']),
    t7_frac_in_nf400          = np.array(t7_results[400]['frac_in']),
    t8_d055_on_nf1_gp         = np.array(t8_results[('D055_on',  1)]['grad_phys']),
    t8_d055_on_nf1_ga         = np.array(t8_results[('D055_on',  1)]['grad_aug']),
    t8_d055_on_nf400_gp       = np.array(t8_results[('D055_on',  400)]['grad_phys']),
    t8_d055_on_nf400_ga       = np.array(t8_results[('D055_on',  400)]['grad_aug']),
    t8_d055_off_nf1_gp        = np.array(t8_results[('D055_off', 1)]['grad_phys']),
    t8_d055_off_nf1_ga        = np.array(t8_results[('D055_off', 1)]['grad_aug']),
    t8_d055_off_nf400_gp      = np.array(t8_results[('D055_off', 400)]['grad_phys']),
    t8_d055_off_nf400_ga      = np.array(t8_results[('D055_off', 400)]['grad_aug']),
    t9_aug_scales             = np.array(AUG_SCALES),
    t9_val_before             = np.array([t9_val_before[k] for k in AUG_SCALES]),
    t9_val_after              = np.array([t9_val_after[k]  for k in AUG_SCALES]),
    t9_ratio                  = np.array([t9_ratio[k]      for k in AUG_SCALES]),
    t10_aug_scales            = np.array(AUG_SCALES),
    t10_val_before            = np.array([t10_val_before[k] for k in AUG_SCALES]),
    t10_val_after             = np.array([t10_val_after[k]  for k in AUG_SCALES]),
    t10_ratio                 = np.array([t10_ratio[k]      for k in AUG_SCALES]),
    t11_val                   = np.array(t11_val),
    t11_gnorm_enc             = np.array(t11_gnorm_enc),
    t11_gnorm_hfn             = np.array(t11_gnorm_hfn),
    t12_val                   = np.array(t12_val),
    t12_gnorm_enc             = np.array(t12_gnorm_enc),
    t12_gnorm_hfn             = np.array(t12_gnorm_hfn),
    t13_x_norms               = np.array(t13_x_norms),
    t13_phys_norms            = np.array(t13_phys_norms),
    t13_aug_norms             = np.array(t13_aug_norms),
    t14_nf_sweep              = np.array(NF_SWEEP_T14),
    t14_gnorm_enc             = np.array([t14_gnorm_enc[n] for n in NF_SWEEP_T14]),
    t14_gnorm_hfn             = np.array([t14_gnorm_hfn[n] for n in NF_SWEEP_T14]),
    t14_val_before            = np.array([t14_val_before[n] for n in NF_SWEEP_T14]),
    t14_val_after             = np.array([t14_val_after[n]  for n in NF_SWEEP_T14]),
)
print(f'\nAll results -> {SAVE_DIR}/')

## ═══════════════════════════════════════════════════════════════════════════
## Timing DB save
## ═══════════════════════════════════════════════════════════════════════════

_total_elapsed = time.time() - _RUN_SCRIPT_START
_run_record = dict(
    timestamp = _datetime.now().isoformat(timespec='seconds'),
    config    = _THIS_CONFIG,
    timings   = _run_timings,
    total     = round(_total_elapsed, 1),
)
try:
    _db_data = {'runs': []}
    if os.path.exists(_TIMING_DB):
        with open(_TIMING_DB) as _f:
            _db_data = _json.load(_f)
    _db_data['runs'].append(_run_record)
    with open(_TIMING_DB, 'w') as _f:
        _json.dump(_db_data, _f, indent=2)
    print(f'[TIMING] Saved to {_TIMING_DB}')
    print(f'[TIMING] Total: {_total_elapsed:.0f}s = {_total_elapsed/60:.1f}min')
except Exception as _e:
    print(f'[TIMING] Warning: could not save timing DB: {_e}')
