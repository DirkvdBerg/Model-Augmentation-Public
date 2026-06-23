"""
diag_nf_lr.py
-------------
Two-stage diagnostic for baseline encoder pipeline validation:

Stage 1: Encoder init quality vs sampling rate
    Sweeps sampling rates. For each rate, builds the encoder and evaluates
    it directly: sliding I/O windows -> batched encoder forward -> NRMS vs
    x_logical. No model simulation. Selects the lowest rate where degradation
    vs native (20 kHz) is acceptable.

Stage 2: nf x lr grid sweep at selected rate
    Trains for N_DIAG_EPOCHS epochs per (nf, lr) combination. After each
    epoch, evaluates encoder state quality directly (same direct forward
    pass, not open-loop simulation). If states degrade for ALL combinations,
    output MSE alone cannot constrain this system's states.

Usage:
    conda run -n GraduationProject python scripts/gantry/encoder-baseline/diagnostics/diag_nf_lr.py
"""

import os
import sys
import json
import time
import numpy as np
import torch
import deepSI
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Project root ---
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

FS_ORIG = 20000

# Stage 1: Encoder init sampling rate sweep
FS_SWEEP = [20000, 4000, 2000, 1000, 500, 400, 200]
# HEURISTIC: max acceptable worst-state degradation ratio vs native rate.
# Auto-selects the lowest passing rate for Stage 2. Inspect the printed
# table and override fs_selected manually if the threshold is too tight/loose.
ENCODER_INIT_MAX_RATIO = 5.0

DTYPE_NP = np.float32
DTYPE_PT = torch.float32

SEED = 42

# HEURISTIC: Jan's rule of thumb for encoder history length
na = 4 * NX_PHYS + 1  # = 25
nb = na
na_right = 1
nb_right = 1

# Fixed hyperparameters (not swept)
HP_FIXED = dict(
    NX_ANN=0,
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=1,       # Validated: downsampling Test B, up_sample=1 sufficient at 400 Hz (NRMS=3.4e-5)
    batch_size=128,
)

# Stage 2: nf/lr diagnostic grid
NF_VALUES = [20, 40, 80, 160, 200]
LR_VALUES = [5e-4, 1e-4, 5e-5]
N_DIAG_EPOCHS = 10
Q2_EARLY_STOP = 10.0  # abort combination if q2 ratio exceeds this

# Data (baseline-v2)
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

# Output
OUT_DIR = os.path.join(
    PROJECT_ROOT, 'simulations', 'gantry_subnet', 'diagnostics',
)
os.makedirs(OUT_DIR, exist_ok=True)


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
    std_x = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
    std_u = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
    u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
    ystd = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    y0 = (Cd.numpy() @ x_mean.flatten()).astype(DTYPE_NP)

    return dict(
        x_mean=x_mean, std_x=std_x, std_u=std_u, u_mean=u_mean,
        ystd=ystd, y0=y0,
        u_all=u_all, y_all=y_all, x_all=x_all,
    )


# =============================================================================
# Build model
# =============================================================================

def build_model(hp, norm, ts):
    """Build interconnect + SSE_Interconnect with linear_encoder_init."""
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN

    x_mean = norm['x_mean']
    std_x = norm['std_x']
    std_u = norm['std_u']
    u_mean = norm['u_mean']
    ystd = norm['ystd']
    y0 = norm['y0']

    Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
    Dd_np = Dd.numpy()

    PHY_IX = np.arange(NX_PHYS)

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

    # Freeze ANN: baseline = system, no mismatch correction needed
    if NX_ANN == 0:
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
            "n_nodes_per_layer": hp['n_nodes_per_layer'],
            "n_hidden_layers": hp['n_hidden_layers'],
        },
    )

    # Manual normalization
    fit_sys.norm.u0 = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0 = y0
    fit_sys.norm.ystd = ystd

    # --- Encoder: linear_encoder_init (Hoekstra 2026 Eq. 16-17) ---
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
# Encoder quality evaluation -- direct forward pass, no simulation
# =============================================================================

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']

# Encoder was built with nb=nb+nb_right, na=na+na_right (see build_model).
# These constants define the window shape expected by encoder.forward().
_NB_WIN = nb + nb_right   # 26
_NA_WIN = na + na_right   # 26
_WIN_START = max(_NB_WIN, _NA_WIN) - 1  # = 25: first timestep with a full window


def evaluate_encoder_direct(encoder, val_u, val_y, val_x, norm):
    """Evaluate encoder quality via direct sliding-window forward pass.

    The encoder is a static map: x_hat = encoder(u_window, y_window).
    For each timestep t >= _WIN_START, build the I/O window, call the encoder,
    compare the output to x_logical[t], and compute per-state NRMS.

    No model simulation is performed -- this isolates encoder quality from
    model propagation error.

    Args:
        encoder: LinearInitEncoderWrapper (from fit_sys.encoder)
        val_u:   (N, nu) raw input array
        val_y:   (N, ny) raw output array
        val_x:   (N, NX_PHYS) ground truth physical states
        norm:    normalization dict from compute_normalization()

    Returns:
        nrms: (NX_PHYS,) per-state NRMS in physical units
    """
    u_mean = norm['u_mean'].flatten()   # (nu,)
    std_u  = norm['std_u'].flatten()    # (nu,)
    y0     = norm['y0']                 # (ny,)
    ystd   = norm['ystd']               # (ny,)
    x_mean = norm['x_mean'].flatten()   # (NX_PHYS,)
    std_x  = norm['std_x'].flatten()    # (NX_PHYS,)

    # Pipeline-normalized inputs: (u - u_mean)/std_u, (y - y0)/ystd
    u_norm = (val_u - u_mean) / std_u  # (N, nu)
    y_norm = (val_y - y0)    / ystd    # (N, ny)

    # Vectorized sliding windows -- no Python loop.
    # sliding_window_view on (N, ch) with window (win, ch) gives (N-win+1, 1, win, ch).
    # After reshape: (N-win+1, win, ch). Window i covers u_norm[i : i+win, :].
    u_wins = np.lib.stride_tricks.sliding_window_view(
        u_norm, (_NB_WIN, nu)).reshape(-1, _NB_WIN, nu)  # (N-_WIN_START, _NB_WIN, nu)
    y_wins = np.lib.stride_tricks.sliding_window_view(
        y_norm, (_NA_WIN, ny)).reshape(-1, _NA_WIN, ny)  # (N-_WIN_START, _NA_WIN, ny)

    # sliding_window_view returns a read-only view; copy before converting to tensor
    u_batch = torch.tensor(u_wins.copy(), dtype=DTYPE_PT)  # (T, _NB_WIN, nu)
    y_batch = torch.tensor(y_wins.copy(), dtype=DTYPE_PT)  # (T, _NA_WIN, ny)

    encoder.eval()
    with torch.no_grad():
        # encoder.forward() flattens internally; output is pipeline-normalized:
        # x_enc = (x_phys - x_mean) / std_x
        x_enc = encoder(u_batch, y_batch).numpy()  # (T, NX_PHYS)

    # Un-normalize to physical units
    x_phys = x_enc * std_x + x_mean  # (T, NX_PHYS)

    # Window i covers u_norm[i:i+_NB_WIN], so the corresponding timestep is
    # t = i + _NB_WIN - 1 = i + _WIN_START. For i=0: t=_WIN_START=25.
    x_gt = val_x[_WIN_START:]         # (N-_WIN_START, NX_PHYS)
    T = min(len(x_phys), len(x_gt))
    x_phys = x_phys[:T]
    x_gt   = x_gt[:T]

    rms_err = np.sqrt(np.mean((x_phys - x_gt) ** 2, axis=0))  # (NX_PHYS,)
    rms_gt  = np.sqrt(np.mean(x_gt ** 2, axis=0))              # (NX_PHYS,)
    return rms_err / (rms_gt + 1e-12)                          # (NX_PHYS,)


# =============================================================================
# Main
# =============================================================================

def main():
    print('=' * 70, flush=True)
    print('Diagnostic: encoder init sweep + nf x lr grid', flush=True)
    print('=' * 70, flush=True)

    # =================================================================
    # STAGE 1: Encoder init quality vs sampling rate
    # =================================================================
    print('\n' + '=' * 70, flush=True)
    print('STAGE 1: Encoder init quality vs sampling rate', flush=True)
    print(f'Rates: {FS_SWEEP}', flush=True)
    print(f'Max ratio threshold: {ENCODER_INIT_MAX_RATIO}x (HEURISTIC -- inspect table)', flush=True)
    print('=' * 70, flush=True)

    sweep_nrms = {}

    for fs in FS_SWEEP:
        d = FS_ORIG // fs
        ts = 1.0 / fs

        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # Load training data (for normalization) and validation data
        train_data = [load_mat(f, d) for f in TRAIN_FILES]
        val_u, val_y, val_x = load_mat(VAL_FILE, d)
        norm = compute_normalization(train_data)

        t0 = time.time()
        model = build_model(HP_FIXED, norm, ts)

        # Direct encoder eval: no init_model, no apply_experiment, no simulation.
        # The encoder is a static map set in build_model -- call it directly.
        nrms = evaluate_encoder_direct(model.encoder, val_u, val_y, val_x, norm)
        elapsed = time.time() - t0
        sweep_nrms[fs] = nrms
        del model

        nrms_str = '  '.join(f'{v:.3e}' for v in nrms)
        print(f'  fs={fs:5d} Hz (D={d:3d}): [{nrms_str}]  ({elapsed:.1f}s)', flush=True)

    # --- Print comparison table ---
    ref_nrms = sweep_nrms[FS_ORIG]
    print(flush=True)
    header = f'  {"fs [Hz]":>8s} |'
    for name in STATE_NAMES:
        header += f' {name + " NRMS":>11s} {name + " ratio":>10s} |'
    header += ' worst ratio'
    print(header, flush=True)
    print('  ' + '-' * (len(header) - 2), flush=True)

    fs_selected = None
    for fs in FS_SWEEP:
        nrms = sweep_nrms[fs]
        ratios = nrms / (ref_nrms + 1e-12)
        worst = np.max(ratios)

        row = f'  {fs:8d} |'
        for i in range(len(STATE_NAMES)):
            row += f' {nrms[i]:11.3e} {ratios[i]:9.2f}x |'
        row += f' {worst:9.2f}x'

        if fs == FS_ORIG:
            row += '  (reference)'
        elif worst <= ENCODER_INIT_MAX_RATIO:
            row += '  OK'
        else:
            row += '  FAIL'

        print(row, flush=True)

    # Select lowest passing rate (ascending sort so we pick the slowest-possible rate)
    for fs in sorted(FS_SWEEP):
        ratios = sweep_nrms[fs] / (ref_nrms + 1e-12)
        if np.max(ratios) <= ENCODER_INIT_MAX_RATIO:
            fs_selected = fs
            break

    if fs_selected is None:
        print('\nERROR: No sampling rate passed the encoder init threshold.', flush=True)
        print('Only the native rate works. Inspect the table above and consider', flush=True)
        print('raising ENCODER_INIT_MAX_RATIO or using FS_NEW=20000 for Stage 2.', flush=True)
        return

    d_sel = FS_ORIG // fs_selected
    ts_sel = 1.0 / fs_selected

    print(f'\nSelected: fs = {fs_selected} Hz '
          f'(lowest rate with worst ratio <= {ENCODER_INIT_MAX_RATIO}x)', flush=True)

    # Save Stage 1 results immediately -- before Stage 2 starts -- so a Stage 2
    # crash does not lose the sweep data.
    sweep_json = {
        'fs_sweep': FS_SWEEP,
        'reference_fs': FS_ORIG,
        'max_ratio_threshold': ENCODER_INIT_MAX_RATIO,
        'selected_fs': fs_selected,
        'results': {
            str(fs): {
                'nrms': {name: float(sweep_nrms[fs][i])
                         for i, name in enumerate(STATE_NAMES)},
                'ratios': {name: float(sweep_nrms[fs][i] / (ref_nrms[i] + 1e-12))
                           for i, name in enumerate(STATE_NAMES)},
            }
            for fs in FS_SWEEP
        },
    }
    stage1_json_path = os.path.join(OUT_DIR, 'diagnostic_stage1_sweep.json')
    with open(stage1_json_path, 'w') as f:
        json.dump(sweep_json, f, indent=2)
    print(f'\nStage 1 saved: {stage1_json_path}', flush=True)

    # Guard: Stage 2 pre-allocates all training windows as one array.
    # At high sampling rates this causes OOM. If the selected rate exceeds
    # FS_STAGE2_MAX, print a warning and stop -- inspect Stage 1 results
    # and raise ENCODER_INIT_MAX_RATIO or accept a higher rate manually.
    FS_STAGE2_MAX = 2000  # HEURISTIC: ~40k windows at 400 Hz is fine; 200k+ OOMs
    if fs_selected > FS_STAGE2_MAX:
        print(f'\nWARNING: fs_selected={fs_selected} Hz exceeds FS_STAGE2_MAX={FS_STAGE2_MAX} Hz.', flush=True)
        print('Stage 2 would OOM pre-allocating training windows at this rate.', flush=True)
        print('Inspect the Stage 1 table above to understand which state is driving', flush=True)
        print('the degradation, then adjust ENCODER_INIT_MAX_RATIO or FS_STAGE2_MAX.', flush=True)
        print('Stopping after Stage 1.', flush=True)
        return

    # =================================================================
    # STAGE 2: nf x lr sweep at selected rate
    # =================================================================
    print('\n' + '=' * 70, flush=True)
    print(f'STAGE 2: nf x lr sweep at fs = {fs_selected} Hz', flush=True)
    print(f'nf values: {NF_VALUES}', flush=True)
    print(f'lr values: {LR_VALUES}', flush=True)
    print(f'Epochs per combination: {N_DIAG_EPOCHS}', flush=True)
    print(f'Grid size: {len(NF_VALUES)} x {len(LR_VALUES)} = '
          f'{len(NF_VALUES) * len(LR_VALUES)} runs', flush=True)
    print('=' * 70, flush=True)

    # --- Load data at selected rate ---
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_data = [load_mat(f, d_sel) for f in TRAIN_FILES]
    val_u, val_y, val_x_logical = load_mat(VAL_FILE, d_sel)

    print(f'\nData at {fs_selected} Hz:', flush=True)
    for fname, (u, y, x) in zip(TRAIN_FILES, train_data):
        print(f'  {fname}: u={u.shape}, y={y.shape}', flush=True)
    print(f'  Val ({VAL_FILE}): u={val_u.shape}, y={val_y.shape}', flush=True)

    norm = compute_normalization(train_data)
    print(f'\nstd_x = {norm["std_x"].flatten()}', flush=True)
    print(f'std_u = {norm["std_u"].flatten()}', flush=True)

    train_list = [
        deepSI.System_data(u=u, y=y, dt=ts_sel)
        for u, y, _ in train_data
    ]
    train_sys_data = deepSI.System_data_list(train_list)
    val_sys_data = deepSI.System_data(u=val_u, y=val_y, dt=ts_sel)

    # --- Sweep ---
    results = {}
    lr_colors = ['tab:red', 'tab:orange', 'tab:blue']
    init_nrms_ref = None  # encoder init NRMS at selected rate (for plot reference line)

    for nf in NF_VALUES:
        for lr in LR_VALUES:
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            fit_sys = build_model(HP_FIXED, norm, ts_sel)
            fit_sys.init_model(
                sys_data=train_sys_data,
                auto_fit_norm=False,
                optimizer_kwargs={'lr': lr},
            )
            fit_sys.hfn.to(DTYPE_PT)

            val_measure = f'{nf}-step-RMS'

            # --- Epoch 0: encoder init quality (direct eval, no simulation) ---
            nrms_0 = evaluate_encoder_direct(
                fit_sys.encoder, val_u, val_y, val_x_logical, norm)
            q2_init_val = float(nrms_0[1])
            state_curves = {name: [float(nrms_0[i])]
                           for i, name in enumerate(STATE_NAMES)}

            if init_nrms_ref is None:
                init_nrms_ref = nrms_0.copy()

            # --- Print header ---
            print('\n' + '=' * 100, flush=True)
            print(f'nf={nf}, lr={lr:.0e}', flush=True)
            print('-' * 100, flush=True)
            header = f'{"epoch":>5s} | {"val_loss":>10s}'
            for name in STATE_NAMES:
                header += f' | {name:>9s}'
            print(header, flush=True)
            print('-' * 100, flush=True)

            # Epoch 0 row
            row = f'{"init":>5s} | {"":>10s}'
            for i in range(len(STATE_NAMES)):
                row += f' | {nrms_0[i]:9.3e}'
            print(row, flush=True)

            # --- Train epoch by epoch ---
            # deepSI fit() interprets epochs as total cumulative count, not additional.
            # Calling fit(epochs=k) trains to epoch k. Pass k=1,2,...,N sequentially.
            t0 = time.time()
            early_stopped = False
            for epoch in range(1, N_DIAG_EPOCHS + 1):
                fit_sys.fit(
                    train_sys_data=train_sys_data,
                    val_sys_data=val_sys_data,
                    batch_size=HP_FIXED['batch_size'],
                    epochs=epoch,
                    auto_fit_norm=False,
                    loss_kwargs={'nf': nf},
                    validation_measure=val_measure,
                    verbose=False,
                )

                # Direct encoder eval after this epoch
                nrms_ep = evaluate_encoder_direct(
                    fit_sys.encoder, val_u, val_y, val_x_logical, norm)
                for i, name in enumerate(STATE_NAMES):
                    state_curves[name].append(float(nrms_ep[i]))

                # Print epoch row
                ep_loss = float(fit_sys.Loss_val[-1])
                row = f'{epoch:5d} | {ep_loss:10.4e}'
                for i in range(len(STATE_NAMES)):
                    row += f' | {nrms_ep[i]:9.3e}'

                # Early stopping: q2 ratio check
                q2_now = float(nrms_ep[1])
                q2_ratio_now = q2_now / q2_init_val if q2_init_val > 0 else float('inf')
                if q2_ratio_now > Q2_EARLY_STOP:
                    row += f'  EARLY STOP (q2 {q2_ratio_now:.1f}x)'
                    print(row, flush=True)
                    early_stopped = True
                    break

                print(row, flush=True)

            elapsed = time.time() - t0

            # Loss curve from deepSI: one entry per training epoch (no epoch-0 entry)
            loss_curve = [float(l) for l in fit_sys.Loss_val]

            # --- Verdict ---
            q2_init = state_curves['q2'][0]
            q2_final = state_curves['q2'][-1]
            q2_ratio = q2_final / q2_init if q2_init > 0 else float('inf')

            if q2_ratio < 0.8:
                verdict = 'BETTER'
            elif q2_ratio < 1.2:
                verdict = 'STABLE'
            else:
                verdict = 'WORSE'

            print(f'verdict: {verdict} (q2 {q2_ratio:.2f}x, '
                  f'{q2_init:.3e} -> {q2_final:.3e}, {elapsed:.1f}s)', flush=True)

            results[(nf, lr)] = {
                'loss_curve': loss_curve,
                'state_curves': state_curves,
                'q2_ratio': q2_ratio,
                'verdict': verdict,
                'elapsed_s': elapsed,
            }

    # =================================================================
    # Summary
    # =================================================================
    print('\n' + '=' * 70, flush=True)
    print('SUMMARY (by state quality)', flush=True)
    print('=' * 70, flush=True)

    better = [(k, v) for k, v in results.items() if v['verdict'] == 'BETTER']
    stable = [(k, v) for k, v in results.items() if v['verdict'] == 'STABLE']
    worse  = [(k, v) for k, v in results.items() if v['verdict'] == 'WORSE']

    if better:
        print(f'\nBETTER ({len(better)}) - states improved:', flush=True)
        for (nf, lr), v in sorted(better, key=lambda x: x[1]['q2_ratio']):
            q2c = v['state_curves']['q2']
            print(f'  nf={nf:3d}, lr={lr:.0e}  '
                  f'q2: {q2c[0]:.3e} -> {q2c[-1]:.3e}  '
                  f'({v["q2_ratio"]:.2f}x)', flush=True)
    if stable:
        print(f'\nSTABLE ({len(stable)}) - states unchanged:', flush=True)
        for (nf, lr), v in stable:
            q2c = v['state_curves']['q2']
            print(f'  nf={nf:3d}, lr={lr:.0e}  '
                  f'q2: {q2c[0]:.3e} -> {q2c[-1]:.3e}  '
                  f'({v["q2_ratio"]:.2f}x)', flush=True)
    if worse:
        print(f'\nWORSE ({len(worse)}) - states degraded:', flush=True)
        for (nf, lr), v in sorted(worse, key=lambda x: x[1]['q2_ratio'],
                                  reverse=True):
            q2c = v['state_curves']['q2']
            print(f'  nf={nf:3d}, lr={lr:.0e}  '
                  f'q2: {q2c[0]:.3e} -> {q2c[-1]:.3e}  '
                  f'({v["q2_ratio"]:.2f}x)', flush=True)

    if not better and not stable:
        print('\nNO combination improved or preserved state quality.', flush=True)
        print('Output MSE alone cannot constrain states for this system.', flush=True)
        print('Consider adding state regularization to the loss.', flush=True)

    # =================================================================
    # Save JSON (both stages)
    # =================================================================
    json_results = {
        'stage1_encoder_init_sweep': sweep_json,
        'stage2_config': {
            'fs_selected': fs_selected,
            'up_sample': HP_FIXED['up_sample'],
            'na': na, 'nb': nb,
            'nf_values': NF_VALUES, 'lr_values': LR_VALUES,
            'n_epochs': N_DIAG_EPOCHS, 'batch_size': HP_FIXED['batch_size'],
            'train_files': TRAIN_FILES, 'val_file': VAL_FILE,
        },
        'stage2_results': {
            f'nf={nf}_lr={lr:.0e}': v for (nf, lr), v in results.items()
        },
    }
    json_path = os.path.join(OUT_DIR, 'diagnostic_nf_lr.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f'\nSaved: {json_path}', flush=True)

    # =================================================================
    # Plot 1: Heatmaps (loss ratio + q2 ratio)
    # =================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: loss ratio (end/start)
    grid_loss = np.zeros((len(NF_VALUES), len(LR_VALUES)))
    for i, nf in enumerate(NF_VALUES):
        for j, lr in enumerate(LR_VALUES):
            lc = results[(nf, lr)]['loss_curve']
            grid_loss[i, j] = lc[-1] / lc[0] if len(lc) >= 2 and lc[0] > 0 else 1.0

    ax = axes[0]
    im = ax.imshow(grid_loss, cmap='RdYlGn_r', aspect='auto', vmin=0.0, vmax=2.0)
    ax.set_xticks(range(len(LR_VALUES)))
    ax.set_xticklabels([f'{lr:.0e}' for lr in LR_VALUES])
    ax.set_yticks(range(len(NF_VALUES)))
    ax.set_yticklabels([str(nf) for nf in NF_VALUES])
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('nf (rollout horizon)')
    ax.set_title('Loss ratio (end / start)')
    for i in range(len(NF_VALUES)):
        for j in range(len(LR_VALUES)):
            val = grid_loss[i, j]
            color = 'white' if val > 1.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: q2 NRMS ratio (final/init)
    grid_q2 = np.zeros((len(NF_VALUES), len(LR_VALUES)))
    for i, nf in enumerate(NF_VALUES):
        for j, lr in enumerate(LR_VALUES):
            grid_q2[i, j] = results[(nf, lr)]['q2_ratio']

    ax = axes[1]
    im = ax.imshow(grid_q2, cmap='RdYlGn_r', aspect='auto', vmin=0.0, vmax=5.0)
    ax.set_xticks(range(len(LR_VALUES)))
    ax.set_xticklabels([f'{lr:.0e}' for lr in LR_VALUES])
    ax.set_yticks(range(len(NF_VALUES)))
    ax.set_yticklabels([str(nf) for nf in NF_VALUES])
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('nf (rollout horizon)')
    ax.set_title('q2 (theta) NRMS ratio (final / init)')
    for i in range(len(NF_VALUES)):
        for j in range(len(LR_VALUES)):
            val = grid_q2[i, j]
            color = 'white' if val > 3.0 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f'Baseline encoder diagnostic (fs={fs_selected} Hz, '
        f'{N_DIAG_EPOCHS} epochs)',
        fontsize=13,
    )
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'diagnostic_nf_lr_heatmap.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {plot_path}', flush=True)

    # =================================================================
    # Plot 2: Loss curves per nf
    # loss_curve has one entry per training epoch (epochs 1..N).
    # x-axis starts at 1 to match the state curve epoch numbering.
    # =================================================================
    fig, axes = plt.subplots(
        len(NF_VALUES), 1, figsize=(10, 3 * len(NF_VALUES)), sharex=True,
    )

    for i, nf in enumerate(NF_VALUES):
        ax = axes[i]
        for j, lr in enumerate(LR_VALUES):
            curve = results[(nf, lr)]['loss_curve']
            # epochs 1..len(curve); range(1, len+1) aligns with state curve x-axis
            ax.plot(range(1, len(curve) + 1), curve, 'o-', color=lr_colors[j],
                    label=f'lr={lr:.0e}', markersize=4, linewidth=1.2)
        ax.set_ylabel(f'nf={nf}\nVal loss')
        ax.set_yscale('log')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Epoch')
    fig.suptitle(f'Loss curves (fs={fs_selected} Hz)', fontsize=13)
    fig.tight_layout()
    curve_path = os.path.join(OUT_DIR, 'diagnostic_nf_lr_curves.png')
    fig.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {curve_path}', flush=True)

    # =================================================================
    # Plot 3: q2 state evolution per nf
    # state_curves has N+1 entries: index 0 = init, indices 1..N = post-epoch.
    # =================================================================
    fig, axes = plt.subplots(
        len(NF_VALUES), 1, figsize=(10, 3 * len(NF_VALUES)), sharex=True,
    )

    for i, nf in enumerate(NF_VALUES):
        ax = axes[i]
        for j, lr in enumerate(LR_VALUES):
            q2_curve = results[(nf, lr)]['state_curves']['q2']
            ax.plot(range(len(q2_curve)), q2_curve, 'o-', color=lr_colors[j],
                    label=f'lr={lr:.0e}', markersize=4, linewidth=1.2)
        # Reference line: encoder init NRMS at selected rate (direct eval)
        if init_nrms_ref is not None:
            ax.axhline(y=float(init_nrms_ref[1]), color='green',
                       linestyle='--', alpha=0.5, linewidth=1,
                       label='init (direct)')
        ax.set_ylabel(f'nf={nf}\nq2 NRMS')
        ax.set_yscale('log')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Epoch (0 = init)')
    fig.suptitle(
        f'q2 (theta) state quality (fs={fs_selected} Hz)', fontsize=13)
    fig.tight_layout()
    state_path = os.path.join(OUT_DIR, 'diagnostic_nf_lr_states.png')
    fig.savefig(state_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {state_path}', flush=True)

    # =================================================================
    # Plot 4: Encoder init quality vs sampling rate
    # =================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    fs_sorted = sorted(FS_SWEEP)
    state_colors = ['tab:blue', 'tab:red', 'tab:green',
                    'tab:cyan', 'tab:orange', 'tab:purple']

    for i, name in enumerate(STATE_NAMES):
        ratios = [sweep_nrms[fs][i] / (ref_nrms[i] + 1e-12)
                  for fs in fs_sorted]
        ax.plot(fs_sorted, ratios, 'o-', color=state_colors[i],
                label=name, markersize=5, linewidth=1.5)

    ax.axhline(y=ENCODER_INIT_MAX_RATIO, color='black', linestyle='--',
               alpha=0.5, linewidth=1, label=f'threshold ({ENCODER_INIT_MAX_RATIO}x)')
    ax.axvline(x=fs_selected, color='green', linestyle=':',
               alpha=0.7, linewidth=1.5, label=f'selected ({fs_selected} Hz)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sampling rate [Hz]')
    ax.set_ylabel('NRMS ratio vs native (20 kHz)')
    ax.set_title('Encoder init quality degradation vs sampling rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(fs_sorted)
    ax.set_xticklabels([str(f) for f in fs_sorted], rotation=45)

    fig.tight_layout()
    sweep_path = os.path.join(OUT_DIR, 'encoder_init_vs_fs.png')
    fig.savefig(sweep_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {sweep_path}', flush=True)

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
