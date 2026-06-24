"""
diag_nf_lr_400hz.py
-------------------
nf x lr grid diagnostic at native 400 Hz initialization and training.

Goal: find which (nf, lr) combinations allow output loss to converge
WITHOUT degrading physical state quality below the native 400 Hz encoder
initialization. No regularization -- pure output MSE training.

Per-epoch tracking: direct sliding-window encoder eval (fast).
Pre/post-training:  apply_experiment (full closed-loop simulation), reporting
both output RMS (y_hat vs y) and state RMS (x_hat vs x_logical) plus NRMS.
Verdict is based on apply_experiment RMS ratio at the best checkpoint.

Two reference lines per state panel:
    green dashed   = 20 kHz native init NRMS (theoretical ceiling)
    orange dotted  = 400 Hz native init NRMS (the floor -- must not fall below this)
    purple dash-dot = analytical baseline (P_inv + finite-diff)

Verdict per (nf, lr) -- observable states q1, q3, dq1, dq3 only:
    BETTER  : worst obs RMS ratio < 0.8
    STABLE  : worst obs RMS ratio < 1.2
    WORSE   : worst obs RMS ratio >= 1.2

Early stopping: EARLY_STOP_PATIENCE consecutive epochs with worst obs NRMS ratio
> EARLY_STOP_RATIO trigger a stop. Single-epoch spikes are tolerated.

Model is saved for BETTER and STABLE combos only.

Usage:
    conda run -n GraduationProject python scripts/gantry/encoder-baseline/diagnostics/diag_nf_lr_400hz.py
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch
import deepSI
from datetime import datetime
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model_augmentation.utils.utils import (
    normalize_linear_ss_matrices, selection_matrix, expansion_matrix,
)
from model_augmentation.utils.torch_nets import (
    zero_init_feed_forward_nn, LinearInitEncoderWrapper,
)
from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect
from model_augmentation.fit_systems.blocks import (
    Gantry_State_Block, Linear_Output_Block, Static_ANN_Block,
)
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize

# =============================================================================
# Configuration
# =============================================================================

NX_PHYS = 6
nu = 3
ny = 3
Y_OP = None  # LPV self-scheduled

FS_ORIG  = 20000   # native data rate
FS_NEW   = 400     # training rate -- passes model discretization validation
D        = FS_ORIG // FS_NEW   # downsample factor = 50
ts       = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32
SEED = 42

# HEURISTIC: Jan's rule of thumb for encoder history length
na = 4 * NX_PHYS + 1   # = 25
nb = na
na_right = 1
nb_right = 1

# Fixed hyperparameters (not swept)
HP_FIXED = dict(
    NX_ANN=0,
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=1,     # validated: downsampling Test B, up_sample=1 sufficient at 400 Hz
    batch_size=128,
)

# nf/lr diagnostic grid
NF_VALUES     = [20, 40, 80, 160, 200]
LR_VALUES     = [5e-4, 1e-4, 5e-5]
N_DIAG_EPOCHS = 10

# Early stopping: EARLY_STOP_PATIENCE consecutive epochs with worst obs ratio
# exceeding EARLY_STOP_RATIO trigger a stop. Resets on any good epoch.
EARLY_STOP_RATIO    = 10.0
EARLY_STOP_PATIENCE = 3

# Observable states (q1, q3, dq1, dq3): indices 0, 2, 3, 5
# q2 (index 1) and dq2 (index 4) excluded from verdict -- unobservable at 400 Hz
OBS_IDX   = [0, 2, 3, 5]
UNOBS_IDX = [1, 4]

# Reference NRMS values from diagnostic_nf_lr.py Stage 1 (already measured)
# 20 kHz native init -- theoretical ceiling
REF_20K = np.array(
    [9.181e-08, 4.614e-02, 1.859e-07, 2.543e-05, 7.003e+00, 1.079e-03],
    dtype=DTYPE_NP,
)
# 400 Hz native init -- the floor: states must not fall below this after training
REF_400 = np.array(
    [5.840e-04, 1.336e+02, 4.586e-04, 5.169e-03, 3.344e+02, 4.633e-02],
    dtype=DTYPE_NP,
)

# Data
TRAJ_DIR = os.path.join(
    PROJECT_ROOT, 'data', 'gantry', 'matlab', 'multisine', 'baseline-v2',
)
TRAIN_FILES = [
    'T1_Y_osc.mat',
    'T2_X_sym_Y_sweep.mat',
    'T3_X_sym_Y000.mat',
    'T4_X_sym_Y030.mat',
    'T5_theta_Y_coupling.mat',
    'T6_lissajous_XY.mat',
    'T7_full_MIMO.mat',
    'T8_multi_amp.mat',
    'T9_Y_sweep_repeated.mat',
    'T10_multi_axis_repeated.mat',
]
VAL_FILE = 'V1_osc_Y025.mat'

OUT_DIR = os.path.join(
    PROJECT_ROOT, 'simulations', 'gantry_subnet', 'diagnostics',
)
os.makedirs(OUT_DIR, exist_ok=True)

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']
PHYS_UNITS  = ['m',  'm',  'm',  'm/s', 'm/s', 'm/s']
OUT_NAMES   = ['y1', 'y2', 'y3']
OUT_UNITS   = ['m',  'm',  'm']

# Window sizes seen by encoder.forward()
_NB_WIN    = nb + nb_right   # 26
_NA_WIN    = na + na_right   # 26
_WIN_START = max(_NB_WIN, _NA_WIN) - 1  # = 25


# =============================================================================
# Data loading
# =============================================================================

def load_mat(filename, downsample=1):
    """Load u, y, x_logical from .mat file, downsample by given factor."""
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = d['u_total'][::downsample].astype(DTYPE_NP)
    y = d['y'][::downsample].astype(DTYPE_NP)
    x_logical = d['x_logical'][::downsample].astype(DTYPE_NP)
    return u, y, x_logical


def compute_normalization(train_data):
    """Compute normalization constants from training data."""
    u_all = np.concatenate([u for u, _, _ in train_data])
    y_all = np.concatenate([y for _, y, _ in train_data])
    x_all = np.concatenate([x for _, _, x in train_data])

    x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
    std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
    std_u  = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
    u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
    ystd   = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    y0     = (Cd.numpy() @ x_mean.flatten()).astype(DTYPE_NP)

    return dict(
        x_mean=x_mean, std_x=std_x, std_u=std_u, u_mean=u_mean,
        ystd=ystd, y0=y0,
        u_all=u_all, y_all=y_all, x_all=x_all,
    )


# =============================================================================
# Build model at 400 Hz (native init)
# =============================================================================

def build_model(hp, norm):
    """Build interconnect + SSE_Interconnect with native 400 Hz encoder init."""
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN

    x_mean = norm['x_mean']
    std_x  = norm['std_x']
    std_u  = norm['std_u']
    u_mean = norm['u_mean']
    ystd   = norm['ystd']
    y0     = norm['y0']

    Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
    Dd_np   = Dd.numpy()
    PHY_IX  = np.arange(NX_PHYS)

    ic = Interconnect(nxd, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=ts,
        up_sample=hp['up_sample'],
    ).to(DTYPE_PT)
    out_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    ic.add_block(phy_block)
    ic.add_block(out_block)

    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=nxd,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=torch.nn.Tanh,
    )
    ic.add_block(ann_block)

    # Freeze ANN: NX_ANN=0 baseline
    for p in ann_block.parameters():
        p.requires_grad = False

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
            'n_nodes_per_layer': hp['n_nodes_per_layer'],
            'n_hidden_layers':   hp['n_hidden_layers'],
        },
    )

    fit_sys.norm.u0   = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0   = y0
    fit_sys.norm.ystd = ystd

    # Native 400 Hz encoder init (reconstructability-based, Hoekstra 2026 Eq. 16-17)
    Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=ts)

    sys_data_with_x = deepSI.System_data(u=norm['u_all'], y=norm['y_all'])
    sys_data_with_x.x = norm['x_all']

    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        Ad, Bd, Cd_dt, Dd_dt, sys_data_with_x)

    phys_encoder = linear_encoder_init(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        flag_linear_only=False,
    )

    fit_sys.encoder = LinearInitEncoderWrapper(
        phys_encoder=phys_encoder,
        nx_ann=NX_ANN,
        nb=nb + nb_right, nu=nu, na=na + na_right, ny=ny,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        u_mean=u_mean, std_u=std_u, y0=y0, ystd=ystd,
        x_mean=x_mean, std_x=std_x,
    ).to(DTYPE_PT)

    return fit_sys


# =============================================================================
# Evaluation: direct encoder forward pass (per-epoch tracking)
# =============================================================================

def evaluate_encoder_direct(encoder, val_u, val_y, val_x, norm):
    """Evaluate encoder quality via direct sliding-window forward pass.

    No model simulation -- isolates encoder quality from model propagation.
    Used for per-epoch state tracking during training.

    Returns: nrms (NX_PHYS,)
    """
    u_mean = norm['u_mean'].flatten()
    std_u  = norm['std_u'].flatten()
    y0     = norm['y0']
    ystd   = norm['ystd']
    x_mean = norm['x_mean'].flatten()
    std_x  = norm['std_x'].flatten()

    u_norm = (val_u - u_mean) / std_u
    y_norm = (val_y - y0)    / ystd

    u_wins = np.lib.stride_tricks.sliding_window_view(
        u_norm, (_NB_WIN, nu)).reshape(-1, _NB_WIN, nu)
    y_wins = np.lib.stride_tricks.sliding_window_view(
        y_norm, (_NA_WIN, ny)).reshape(-1, _NA_WIN, ny)

    u_batch = torch.tensor(u_wins.copy(), dtype=DTYPE_PT)
    y_batch = torch.tensor(y_wins.copy(), dtype=DTYPE_PT)

    encoder.eval()
    with torch.no_grad():
        x_enc = encoder(u_batch, y_batch).numpy()

    x_phys = x_enc * std_x + x_mean
    x_gt   = val_x[_WIN_START:]
    T      = min(len(x_phys), len(x_gt))
    x_phys = x_phys[:T]
    x_gt   = x_gt[:T]

    rms_err = np.sqrt(np.mean((x_phys - x_gt) ** 2, axis=0))
    rms_gt  = np.sqrt(np.mean(x_gt ** 2, axis=0))
    return rms_err / (rms_gt + 1e-12)


# =============================================================================
# Evaluation: apply_experiment (full closed-loop simulation)
# =============================================================================

def compute_rms_error(a, b):
    """Per-channel RMS error (same units as input)."""
    return np.sqrt(np.mean((a - b) ** 2, axis=0))


def compute_analytical_baseline(y, x_logical):
    """Analytical state estimates: P_inv for positions, backward FD for velocities.

    Returns nrms (NX_PHYS,), rms (NX_PHYS,), x_analytical (N, NX_PHYS).
    """
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)

    # THEORY: q_logical = P_inv @ y_stage (measurement equation)
    pos = (P_inv_T @ y.T).T  # (N, 3)

    # HEURISTIC: backward finite difference for velocity, O(Ts) accurate
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * FS_NEW
    vel[0]  = vel[1]

    x_analytical = np.hstack([pos, vel])  # (N, 6)

    rms    = compute_rms_error(x_analytical, x_logical)
    rms_gt = np.sqrt(np.mean(x_logical ** 2, axis=0))
    nrms   = rms / (rms_gt + 1e-12)

    return nrms, rms, x_analytical


def evaluate_apply_experiment(fit_sys, val_sys_data, x_logical_val, norm):
    """Full closed-loop simulation via apply_experiment.

    Returns:
        nrms_state  (NX_PHYS,)  -- per-state NRMS vs x_logical
        rms_state   (NX_PHYS,)  -- per-state RMS in physical units
        rms_output  (ny,)       -- per-channel output RMS, y_hat vs y_measured
        x_enc_phys  (T, NX_PHYS)
        x_gt        (T, NX_PHYS)
        cheat_n     int
    """
    x_mean = norm['x_mean']
    std_x  = norm['std_x']

    fit_sys.eval()
    fit_sys.hfn.reset_saved_signals()
    sim_result = fit_sys.apply_experiment(val_sys_data)
    cheat_n    = sim_result.cheat_n

    # States: saved_output_signals shape (total_dim, T), first NX_PHYS = normalized physical
    x_enc_norm = np.array(fit_sys.hfn.saved_output_signals)[:NX_PHYS, :]  # (6, T)
    x_enc_phys = (x_enc_norm * std_x + x_mean).T                          # (T, 6)

    x_gt = x_logical_val[cheat_n:]
    T    = min(len(x_enc_phys), len(x_gt))
    x_enc_phys = x_enc_phys[:T]
    x_gt       = x_gt[:T]

    rms_state  = compute_rms_error(x_enc_phys, x_gt)
    rms_gt     = np.sqrt(np.mean(x_gt ** 2, axis=0))
    nrms_state = rms_state / (rms_gt + 1e-12)

    # Output RMS: y_hat from simulation vs y_measured
    y_hat = np.array(sim_result.y)
    y_ref = np.array(val_sys_data.y)[cheat_n:]
    T_y   = min(len(y_hat), len(y_ref))
    rms_output = compute_rms_error(y_hat[:T_y], y_ref[:T_y])

    return nrms_state, rms_state, rms_output, x_enc_phys, x_gt, cheat_n


# =============================================================================
# Verdict
# =============================================================================

def compute_verdict(arr_init, arr_best):
    """BETTER / STABLE / WORSE based on observable state ratios.

    Works with either NRMS or RMS arrays (ratio is identical since
    nrms_best/nrms_init = rms_best/rms_init when x_gt is the same).

    Returns: (verdict_str, worst_ratio)
    """
    ratios = arr_best[OBS_IDX] / (arr_init[OBS_IDX] + 1e-12)
    worst  = float(np.max(ratios))
    if worst < 0.8:
        return 'BETTER', worst
    elif worst < 1.2:
        return 'STABLE', worst
    else:
        return 'WORSE', worst


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--task_idx', type=int, default=None,
        help='Grid task index (0 to N_combos-1) for SLURM array jobs. '
             'Omit to run all combinations sequentially.',
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')

    print('=' * 70, flush=True)
    print('Diagnostic: nf x lr grid at native 400 Hz init (no regularization)', flush=True)
    print('=' * 70, flush=True)
    print(f'  run_id        = {run_id}', flush=True)
    print(f'  FS_NEW        = {FS_NEW} Hz  (D = {D})', flush=True)
    print(f'  NF_VALUES     = {NF_VALUES}', flush=True)
    print(f'  LR_VALUES     = {LR_VALUES}', flush=True)
    print(f'  N_DIAG_EPOCHS = {N_DIAG_EPOCHS}', flush=True)
    print(f'  Early stop    = {EARLY_STOP_PATIENCE} consecutive epochs > {EARLY_STOP_RATIO:.0f}x', flush=True)
    print(f'  Grid: {len(NF_VALUES)} x {len(LR_VALUES)} = '
          f'{len(NF_VALUES) * len(LR_VALUES)} runs', flush=True)

    # ── SLURM array / single-task filtering ──────────────────────────────────
    args = parse_args()
    all_combos = [(nf, lr) for nf in NF_VALUES for lr in LR_VALUES]
    if args.task_idx is not None:
        if not (0 <= args.task_idx < len(all_combos)):
            raise ValueError(
                f'--task_idx {args.task_idx} out of range [0, {len(all_combos) - 1}]'
            )
        _task_nf, _task_lr = all_combos[args.task_idx]
        run_set = {(_task_nf, _task_lr)}
        single_task = True
        print(f'  [SLURM task {args.task_idx}]  nf={_task_nf}, lr={_task_lr:.0e}',
              flush=True)
    else:
        run_set = None   # run everything
        single_task = False

    # Print reference NRMS values
    print(flush=True)
    print('Reference NRMS (from Stage 1 of diagnostic_nf_lr.py):', flush=True)
    header = f'  {"":>25s} |'
    for name in STATE_NAMES:
        header += f' {name:>11s} |'
    print(header, flush=True)
    ref_row_20k = f'  {"20kHz native (ceiling)":>25s} |'
    ref_row_400 = f'  {"400Hz native (floor)":>25s} |'
    for i in range(NX_PHYS):
        ref_row_20k += f' {REF_20K[i]:11.3e} |'
        ref_row_400 += f' {REF_400[i]:11.3e} |'
    print(ref_row_20k, flush=True)
    print(ref_row_400, flush=True)
    print(f'  Observable states (verdict): q1, q3, dq1, dq3', flush=True)
    print(f'  Unobservable (tracked only): q2, dq2', flush=True)

    # =========================================================================
    # Load data at 400 Hz
    # =========================================================================
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f'\nLoading data at {FS_NEW} Hz (D={D})...', flush=True)
    train_data = [load_mat(f, D) for f in TRAIN_FILES]
    val_u, val_y, val_x_logical = load_mat(VAL_FILE, D)

    print(f'  Samples per training file: {train_data[0][0].shape[0]}', flush=True)
    print(f'  Validation samples:        {val_u.shape[0]}', flush=True)

    norm = compute_normalization(train_data)

    train_list = [
        deepSI.System_data(u=u, y=y, dt=ts)
        for u, y, _ in train_data
    ]
    train_sys_data = deepSI.System_data_list(train_list)
    val_sys_data   = deepSI.System_data(u=val_u, y=val_y, dt=ts)

    # =========================================================================
    # Analytical baseline (once, same val data for all combos)
    # =========================================================================
    print(f'\n--- Analytical baseline (P_inv + FD) on {VAL_FILE} ---', flush=True)
    nrms_ana, rms_ana, _ = compute_analytical_baseline(val_y, val_x_logical)
    print(f'  {"State":<6s}  {"NRMS":>12s}  {"RMS":>12s}  {"Unit":<5s}', flush=True)
    print(f'  {"-"*6}  {"-"*12}  {"-"*12}  {"-"*5}', flush=True)
    for i, name in enumerate(STATE_NAMES):
        flag = '*' if i in UNOBS_IDX else ' '
        print(f'  {name:<6s}{flag} {nrms_ana[i]:>12.4e}  {rms_ana[i]:>12.4e}  {PHYS_UNITS[i]:<5s}',
              flush=True)
    print(f'  (* = unobservable at 400 Hz)', flush=True)

    # =========================================================================
    # Sweep
    # =========================================================================
    results   = {}
    lr_colors = ['tab:red', 'tab:orange', 'tab:blue']

    for nf in NF_VALUES:
        for lr in LR_VALUES:
            if run_set is not None and (nf, lr) not in run_set:
                continue
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            fit_sys = build_model(HP_FIXED, norm)
            fit_sys.init_model(
                sys_data=train_sys_data,
                auto_fit_norm=False,
                optimizer_kwargs={'lr': lr},
            )
            fit_sys.hfn.to(DTYPE_PT)

            # --- Pre-training: direct eval (init reference for per-epoch plot) ---
            nrms_0 = evaluate_encoder_direct(
                fit_sys.encoder, val_u, val_y, val_x_logical, norm)
            state_curves = {name: [float(nrms_0[i])]
                            for i, name in enumerate(STATE_NAMES)}

            # --- Pre-training: apply_experiment (init baseline for verdict table) ---
            (nrms_init_ae, rms_init_ae, output_rms_init,
             _, _, _) = evaluate_apply_experiment(
                fit_sys, val_sys_data, val_x_logical, norm)

            # Print header
            print('\n' + '=' * 120, flush=True)
            print(f'nf={nf}, lr={lr:.0e}', flush=True)
            print('-' * 120, flush=True)
            hdr = f'{"epoch":>5s} | {"train_loss":>11s} | {"val_loss":>10s}'
            for name in STATE_NAMES:
                marker = '*' if STATE_NAMES.index(name) in UNOBS_IDX else ''
                hdr += f' | {name + marker:>9s}'
            print(hdr, flush=True)
            print(f'  (* = unobservable, excluded from verdict)', flush=True)
            print('-' * 120, flush=True)

            # Init row (direct eval)
            row = f'{"init":>5s} | {"":>11s} | {"":>10s}'
            for i in range(NX_PHYS):
                row += f' | {nrms_0[i]:9.3e}'
            print(row, flush=True)

            # --- Training loop: manual loss collection + 3-consecutive early stop ---
            t0 = time.time()
            early_stopped    = False
            consecutive_bad  = 0
            loss_val_history   = []
            loss_train_history = []

            for epoch in range(1, N_DIAG_EPOCHS + 1):
                fit_sys.fit(
                    train_sys_data=train_sys_data,
                    val_sys_data=val_sys_data,
                    batch_size=HP_FIXED['batch_size'],
                    epochs=epoch,
                    auto_fit_norm=False,
                    loss_kwargs={'nf': nf},
                    validation_measure=f'{nf}-step-RMS',
                    verbose=False,
                )

                ep_val_loss   = float(fit_sys.Loss_val[-1])
                ep_train_loss = float(fit_sys.Loss_train[-1]) if len(fit_sys.Loss_train) > 0 else None
                loss_val_history.append(ep_val_loss)
                loss_train_history.append(ep_train_loss)

                nrms_ep = evaluate_encoder_direct(
                    fit_sys.encoder, val_u, val_y, val_x_logical, norm)
                for i, name in enumerate(STATE_NAMES):
                    state_curves[name].append(float(nrms_ep[i]))

                row = f'{epoch:5d} | {ep_train_loss:11.4e} | {ep_val_loss:10.4e}'
                for i in range(NX_PHYS):
                    row += f' | {nrms_ep[i]:9.3e}'

                # 3-consecutive early stopping on observable states
                obs_ratios  = nrms_ep[OBS_IDX] / (nrms_0[OBS_IDX] + 1e-12)
                worst_ratio = float(np.max(obs_ratios))
                worst_state = STATE_NAMES[OBS_IDX[int(np.argmax(obs_ratios))]]
                if worst_ratio > EARLY_STOP_RATIO:
                    consecutive_bad += 1
                else:
                    consecutive_bad = 0

                if consecutive_bad >= EARLY_STOP_PATIENCE:
                    row += (f'  EARLY STOP ({worst_state} {worst_ratio:.1f}x,'
                            f' {consecutive_bad} consec.)')
                    print(row, flush=True)
                    early_stopped = True
                    break

                print(row, flush=True)

            elapsed = time.time() - t0

            # --- Post-training: load best checkpoint, then apply_experiment ---
            fit_sys.checkpoint_load_system(name='_best')

            (nrms_best_ae, rms_best_ae, output_rms_best,
             _, _, _) = evaluate_apply_experiment(
                fit_sys, val_sys_data, val_x_logical, norm)

            # Verdict from apply_experiment RMS ratio at best checkpoint
            verdict, worst = compute_verdict(rms_init_ae, rms_best_ae)

            # Per-state ratio string for printout
            ratio_str = '  '.join(
                f'{STATE_NAMES[i]}:{rms_best_ae[i] / (rms_init_ae[i] + 1e-12):.2f}x'
                for i in OBS_IDX
            )
            print(f'\n  verdict: {verdict} (worst obs ratio {worst:.2f}x)  '
                  f'[{ratio_str}]  ({elapsed:.1f}s)', flush=True)

            # Summary table: init vs best vs analytical (apply_experiment)
            print(f'  {"State":<6s}  {"NRMS init":>12s}  {"NRMS best":>12s}'
                  f'  {"ratio":>7s}  {"RMS init":>12s}  {"RMS best":>12s}'
                  f'  {"RMS analyt.":>12s}  {"Unit":<5s}', flush=True)
            print(f'  {"-"*6}  {"-"*12}  {"-"*12}  {"-"*7}'
                  f'  {"-"*12}  {"-"*12}  {"-"*12}  {"-"*5}', flush=True)
            for i, name in enumerate(STATE_NAMES):
                flag  = '*' if i in UNOBS_IDX else ' '
                ratio = rms_best_ae[i] / (rms_init_ae[i] + 1e-12)
                print(f'  {name:<6s}{flag} {nrms_init_ae[i]:>12.4e}  {nrms_best_ae[i]:>12.4e}'
                      f'  {ratio:>6.2f}x  {rms_init_ae[i]:>12.4e}  {rms_best_ae[i]:>12.4e}'
                      f'  {rms_ana[i]:>12.4e}  {PHYS_UNITS[i]:<5s}', flush=True)
            print(f'  (* unobservable at 400 Hz)', flush=True)

            print(f'  Output RMS (apply_experiment):', flush=True)
            for i in range(ny):
                print(f'    {OUT_NAMES[i]}: init={output_rms_init[i]:.4e}  '
                      f'best={output_rms_best[i]:.4e}  {OUT_UNITS[i]}', flush=True)

            # --- Save model (BETTER or STABLE only) ---
            model_path = None
            if verdict in ('BETTER', 'STABLE'):
                model_fname = f'diag_encoder_nf{nf}_lr{lr:.0e}_{run_id}'
                model_path  = os.path.join(OUT_DIR, model_fname)
                fit_sys.save_system(model_path)
                print(f'  Saved model ({verdict}): {model_path}', flush=True)

            results[(nf, lr)] = {
                # Training curves (manually collected, one value per epoch)
                'loss_val':    loss_val_history,
                'loss_train':  loss_train_history,
                # Per-epoch direct NRMS (epoch 0 = init)
                'state_curves': state_curves,
                # apply_experiment init
                'nrms_init_ae':    {n: float(nrms_init_ae[i])   for i, n in enumerate(STATE_NAMES)},
                'rms_init_ae':     {n: float(rms_init_ae[i])    for i, n in enumerate(STATE_NAMES)},
                'output_rms_init': {n: float(output_rms_init[i]) for i, n in enumerate(OUT_NAMES)},
                # apply_experiment best checkpoint
                'nrms_best_ae':    {n: float(nrms_best_ae[i])   for i, n in enumerate(STATE_NAMES)},
                'rms_best_ae':     {n: float(rms_best_ae[i])    for i, n in enumerate(STATE_NAMES)},
                'output_rms_best': {n: float(output_rms_best[i]) for i, n in enumerate(OUT_NAMES)},
                # verdict + metadata
                'verdict':         verdict,
                'worst_obs_ratio': worst,
                'early_stopped':   early_stopped,
                'elapsed_s':       elapsed,
                'model_path':      model_path,
            }

    # =========================================================================
    # Single-task (SLURM array) path: save partial JSON and exit
    # =========================================================================
    if single_task:
        nf, lr = all_combos[args.task_idx]
        partial = {
            'task_idx': args.task_idx,
            'nf':  nf,
            'lr':  lr,
            'key': f'nf={nf}_lr={lr:.0e}',
            'run_id': run_id,
            'config': {
                'fs_new': FS_NEW, 'ts': ts,
                'nf_values': NF_VALUES, 'lr_values': LR_VALUES,
                'n_epochs': N_DIAG_EPOCHS,
                'early_stop_ratio':    EARLY_STOP_RATIO,
                'early_stop_patience': EARLY_STOP_PATIENCE,
                'na': na, 'nb': nb, 'na_right': na_right, 'nb_right': nb_right,
                'obs_states':   [STATE_NAMES[i] for i in OBS_IDX],
                'unobs_states': [STATE_NAMES[i] for i in UNOBS_IDX],
                'train_files': TRAIN_FILES, 'val_file': VAL_FILE,
            },
            'analytical': {
                'nrms': {n: float(nrms_ana[i]) for i, n in enumerate(STATE_NAMES)},
                'rms':  {n: float(rms_ana[i])  for i, n in enumerate(STATE_NAMES)},
            },
            'result': results[(nf, lr)],
        }
        partial_path = os.path.join(
            OUT_DIR, f'diagnostic_nf_lr_400hz_task_{args.task_idx:02d}.json')
        with open(partial_path, 'w') as f:
            json.dump(partial, f, indent=2)
        print(f'\nSaved partial: {partial_path}', flush=True)
        print('Done (single task).', flush=True)
        return

    # =========================================================================
    # Summary
    # =========================================================================
    print('\n' + '=' * 70, flush=True)
    print('SUMMARY', flush=True)
    print('=' * 70, flush=True)

    for category in ['BETTER', 'STABLE', 'WORSE']:
        matches = [(k, v) for k, v in results.items() if v['verdict'] == category]
        if not matches:
            continue
        print(f'\n{category} ({len(matches)}):', flush=True)
        for (nf, lr), v in sorted(matches, key=lambda x: x[1]['worst_obs_ratio']):
            ratio_detail = '  '.join(
                f'{STATE_NAMES[i]}:{v["rms_best_ae"][STATE_NAMES[i]] / (v["rms_init_ae"][STATE_NAMES[i]] + 1e-12):.2f}x'
                for i in OBS_IDX
            )
            print(f'  nf={nf:3d} lr={lr:.0e}  worst={v["worst_obs_ratio"]:.2f}x  '
                  f'{ratio_detail}', flush=True)

    if all(v['verdict'] == 'WORSE' for v in results.values()):
        print('\nNO combination preserved state quality.', flush=True)
        print('Output MSE alone cannot constrain states. Consider regularization.',
              flush=True)

    # =========================================================================
    # Save combined JSON (serial mode only; SLURM uses partial JSONs)
    # =========================================================================
    json_out = {
        'run_id': run_id,
        'config': {
            'fs_new': FS_NEW, 'ts': ts,
            'nf_values': NF_VALUES, 'lr_values': LR_VALUES,
            'n_epochs': N_DIAG_EPOCHS,
            'early_stop_ratio':    EARLY_STOP_RATIO,
            'early_stop_patience': EARLY_STOP_PATIENCE,
            'na': na, 'nb': nb, 'na_right': na_right, 'nb_right': nb_right,
            'obs_states':   [STATE_NAMES[i] for i in OBS_IDX],
            'unobs_states': [STATE_NAMES[i] for i in UNOBS_IDX],
            'train_files': TRAIN_FILES, 'val_file': VAL_FILE,
        },
        'references': {
            '20khz_native': {n: float(REF_20K[i]) for i, n in enumerate(STATE_NAMES)},
            '400hz_native': {n: float(REF_400[i]) for i, n in enumerate(STATE_NAMES)},
            'analytical': {
                'nrms': {n: float(nrms_ana[i]) for i, n in enumerate(STATE_NAMES)},
                'rms':  {n: float(rms_ana[i])  for i, n in enumerate(STATE_NAMES)},
            },
        },
        'results': {
            f'nf={nf}_lr={lr:.0e}': v
            for (nf, lr), v in results.items()
        },
    }
    json_path = os.path.join(OUT_DIR, f'diagnostic_nf_lr_400hz_{run_id}.json')
    with open(json_path, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f'\nSaved: {json_path}', flush=True)

    # =========================================================================
    # Plot 1: Heatmaps -- loss ratio and worst observable state ratio (ae)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    grid_loss = np.full((len(NF_VALUES), len(LR_VALUES)), np.nan)
    grid_obs  = np.full((len(NF_VALUES), len(LR_VALUES)), np.nan)
    for i, nf in enumerate(NF_VALUES):
        for j, lr in enumerate(LR_VALUES):
            if (nf, lr) not in results:
                continue
            v  = results[(nf, lr)]
            lc = v['loss_val']
            grid_loss[i, j] = lc[-1] / lc[0] if len(lc) >= 2 and lc[0] > 0 else 1.0
            # Obs-state ratio from apply_experiment (best checkpoint)
            rms_i = np.array([v['rms_init_ae'][STATE_NAMES[k]] for k in OBS_IDX])
            rms_b = np.array([v['rms_best_ae'][STATE_NAMES[k]] for k in OBS_IDX])
            grid_obs[i, j] = float(np.max(rms_b / (rms_i + 1e-12)))

    for ax, grid, title, vmax in zip(
        axes,
        [grid_loss, grid_obs],
        ['Val loss ratio (end / start)',
         'Worst obs-state RMS ratio (best / init)\nq1, q3, dq1, dq3  [apply_experiment]'],
        [2.0, 5.0],
    ):
        im = ax.imshow(grid, cmap='RdYlGn_r', aspect='auto', vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(LR_VALUES)))
        ax.set_xticklabels([f'{lr:.0e}' for lr in LR_VALUES])
        ax.set_yticks(range(len(NF_VALUES)))
        ax.set_yticklabels([str(nf) for nf in NF_VALUES])
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('nf (rollout horizon)')
        ax.set_title(title)
        for ii in range(len(NF_VALUES)):
            for jj in range(len(LR_VALUES)):
                val = grid[ii, jj]
                if np.isnan(val):
                    ax.text(jj, ii, 'N/A', ha='center', va='center',
                            fontsize=9, color='gray')
                    continue
                color = 'white' if val > vmax * 0.7 else 'black'
                ax.text(jj, ii, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f'Baseline encoder diagnostic ({FS_NEW} Hz native init, {N_DIAG_EPOCHS} epochs)'
        f'\nrun_id: {run_id}',
        fontsize=12,
    )
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, f'diagnostic_nf_lr_400hz_heatmap_{run_id}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {plot_path}', flush=True)

    # =========================================================================
    # Plot 2: Loss curves per nf (val + train)
    # =========================================================================
    fig, axes = plt.subplots(len(NF_VALUES), 1,
                             figsize=(10, 3 * len(NF_VALUES)), sharex=True)

    for i, nf in enumerate(NF_VALUES):
        ax = axes[i]
        for j, lr in enumerate(LR_VALUES):
            if (nf, lr) not in results:
                continue
            val_curve   = results[(nf, lr)]['loss_val']
            train_curve = results[(nf, lr)]['loss_train']
            epochs = list(range(1, len(val_curve) + 1))
            ax.plot(epochs, val_curve, 'o-',
                    color=lr_colors[j], label=f'val lr={lr:.0e}',
                    markersize=4, linewidth=1.2)
            if train_curve and any(l is not None for l in train_curve):
                train_plot = [l if l is not None else np.nan for l in train_curve]
                ax.plot(epochs, train_plot, '--',
                        color=lr_colors[j], label=f'train lr={lr:.0e}',
                        markersize=2, linewidth=0.8, alpha=0.6)
        ax.set_ylabel(f'nf={nf}\nLoss')
        ax.set_yscale('log')
        ax.legend(fontsize=6, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Epoch')
    fig.suptitle(f'Loss curves ({FS_NEW} Hz native init)  run_id: {run_id}', fontsize=12)
    fig.tight_layout()
    curve_path = os.path.join(OUT_DIR, f'diagnostic_nf_lr_400hz_curves_{run_id}.png')
    fig.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {curve_path}', flush=True)

    # =========================================================================
    # Plot 3: All 6 state NRMS vs epoch (direct eval, training trajectory)
    # Three reference lines:
    #   green dashed    = 20 kHz native init (ceiling)
    #   orange dotted   = 400 Hz native init (floor)
    #   purple dash-dot = analytical baseline (P_inv + FD)
    # =========================================================================
    nf_colors = plt.cm.tab10(np.linspace(0, 0.5, len(NF_VALUES)))
    lr_styles = ['-', '--', ':']

    fig, axes = plt.subplots(NX_PHYS, 1,
                             figsize=(12, 3 * NX_PHYS), sharex=True)

    for si, name in enumerate(STATE_NAMES):
        ax = axes[si]
        for ni, nf in enumerate(NF_VALUES):
            for li, lr in enumerate(LR_VALUES):
                if (nf, lr) not in results:
                    continue
                sc = results[(nf, lr)]['state_curves'][name]
                ax.plot(range(len(sc)), sc,
                        color=nf_colors[ni], linestyle=lr_styles[li],
                        marker='o', markersize=2, linewidth=1.0,
                        label=f'nf={nf},lr={lr:.0e}')

        ax.axhline(y=float(REF_20K[si]), color='green', linestyle='--',
                   linewidth=1.2, alpha=0.8, label='20kHz ceiling')
        ax.axhline(y=float(REF_400[si]), color='orange', linestyle=':',
                   linewidth=1.2, alpha=0.8, label='400Hz floor')
        ax.axhline(y=float(nrms_ana[si]), color='purple', linestyle='-.',
                   linewidth=1.2, alpha=0.8, label='analytical (P_inv+FD)')

        unobs_tag = '  [unobservable]' if si in UNOBS_IDX else ''
        ax.set_ylabel(f'{name}\nNRMS{unobs_tag}')
        ax.set_yscale('log')
        ax.legend(fontsize=5, loc='upper right', ncol=3)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Epoch (0 = init)')
    fig.suptitle(
        f'All state NRMS vs epoch (direct eval, training trajectory)\n'
        f'({FS_NEW} Hz native init, no regularization)  run_id: {run_id}',
        fontsize=11,
    )
    fig.tight_layout()
    state_path = os.path.join(OUT_DIR, f'diagnostic_nf_lr_400hz_states_{run_id}.png')
    fig.savefig(state_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {state_path}', flush=True)

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
