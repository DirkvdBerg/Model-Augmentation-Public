"""
diag_lambda.py
--------------
Two-part diagnostic for the baseline encoder with physical state regularization.

Part 1 -- Mismatch measurement (no training):
    Builds the encoder at 20 kHz (native rate, best init quality).
    Evaluates it on 20 kHz validation data  -> reference NRMS per state.
    Evaluates the SAME weights on 400 Hz validation data -> mismatch NRMS.
    The ratio between these two rows shows what the rate mismatch costs per state,
    before any training has happened.

Part 2 -- Lambda sweep (20 kHz init, 400 Hz training):
    Uses the 20 kHz encoder weights as initialization for 400 Hz training.
    Sweeps the regularization weight lambda on physical state deviation from a
    frozen copy of the 20 kHz encoder. lambda=0 reproduces the known degradation
    (zero gradient signal for weakly observable states). Higher lambda pins the
    encoder near its initialization. Tracks per-epoch output loss AND all 6
    physical state NRMS to find the minimum lambda that prevents state degradation
    without preventing output loss convergence.

Usage:
    conda run -n GraduationProject python scripts/gantry/encoder-baseline/diagnostics/diag_lambda.py
"""

import os
import sys
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import deepSI
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
from model_augmentation.systems.gantry_ss import Cd, Dd
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize

# =============================================================================
# Configuration
# =============================================================================

NX_PHYS = 6
nu = 3
ny = 3
Y_OP = None

FS_ORIG  = 20000  # encoder init rate -- best state reconstruction quality
FS_TRAIN = 400    # training rate -- passes model discretization validation

DTYPE_NP = np.float32
DTYPE_PT = torch.float32
SEED = 42

# HEURISTIC: Jan's rule of thumb for encoder history length
na = 4 * NX_PHYS + 1  # = 25
nb = na
na_right = 1
nb_right = 1

# Window sizes seen by encoder.forward()
_NB_WIN   = nb + nb_right   # 26
_NA_WIN   = na + na_right   # 26
_WIN_START = max(_NB_WIN, _NA_WIN) - 1  # = 25: first timestep with a full window

# Fixed model hyperparameters
HP = dict(
    NX_ANN=0,
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=1,     # validated: downsampling Test B, up_sample=1 sufficient at 400 Hz
    batch_size=128,
)

# Training hyperparameters -- fixed here, not swept
# HEURISTIC: nf ~ 0.1 s / Ts = 0.1 * 400 = 40 (Jan's rule)
# HEURISTIC: lr = 1e-4, conservative end of Jan's 1e-4 to 5e-4 range
NF       = 40
LR       = 1e-4
N_EPOCHS = 20

# Lambda sweep: 0 = pure output MSE (known degradation), increasing lambda pins states
LAMBDA_VALUES = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

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


# =============================================================================
# Data loading and normalization
# =============================================================================

def load_mat(filename, downsample=1):
    """Load u, y, x_logical from .mat file, downsample by given factor."""
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = d['u_total'][::downsample].astype(DTYPE_NP)
    y = d['y'][::downsample].astype(DTYPE_NP)
    x_logical = d['x_logical'][::downsample].astype(DTYPE_NP)
    return u, y, x_logical


def compute_normalization(train_data):
    """Compute normalization constants from training trajectories."""
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
# Regularized SSE_Interconnect
# =============================================================================

class RegularizedSSE_Interconnect(SSE_Interconnect):
    """SSE_Interconnect with physical state regularization loss.

    Adds lambda_reg * MSE(x_enc[:NX_PHYS], x_frozen[:NX_PHYS]) to the output
    MSE loss, where x_frozen is a parameter-frozen copy of the 20 kHz encoder.

    Only the physical states are regularized. ANN augmentation states (if any)
    are left free -- they must learn residual dynamics without being pinned to
    a physics-based reference.
    """

    def __init__(self, *args, frozen_encoder=None, lambda_reg=0.0,
                 nx_phys=NX_PHYS, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_encoder = frozen_encoder
        self.lambda_reg = lambda_reg
        self.nx_phys = nx_phys

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        # Encoder produces the initial state for the nf-step rollout
        x = self.encoder(uhist, yhist)

        # nf-step output MSE rollout
        # Use x_roll to avoid shadowing x (needed below for regularization)
        errors = []
        x_roll = x
        for y_step, u_step in zip(
            torch.transpose(yfuture, 0, 1),
            torch.transpose(ufuture, 0, 1),
        ):
            y_hat, x_roll = self.hfn(x_roll, u_step)
            errors.append(nn.functional.mse_loss(y_step, y_hat))
        loss_mse = torch.mean(torch.stack(errors))

        # Physical state regularization: pin x[:NX_PHYS] near frozen 20 kHz init
        if self.lambda_reg > 0.0 and self.frozen_encoder is not None:
            with torch.no_grad():
                x_frozen = self.frozen_encoder(uhist, yhist)
            loss_reg = nn.functional.mse_loss(
                x[:, :self.nx_phys],
                x_frozen[:, :self.nx_phys],
            )
            return loss_mse + self.lambda_reg * loss_reg

        return loss_mse


# =============================================================================
# Build encoder (20 kHz only -- no model needed)
# =============================================================================

def build_encoder_20khz(norm):
    """Build LinearInitEncoderWrapper using 20 kHz discrete model matrices.

    The reconstructability-based weights (Wb_psi_y, Wb_psi_u) are computed
    from the 20 kHz observability matrix. No model or interconnect is built --
    only the encoder, so it can be evaluated independently and later copied
    into the 400 Hz training model.
    """
    ts = 1.0 / FS_ORIG

    x_mean = norm['x_mean']
    std_x  = norm['std_x']
    std_u  = norm['std_u']
    u_mean = norm['u_mean']
    ystd   = norm['ystd']
    y0     = norm['y0']

    Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=ts)

    sys_data_with_x = deepSI.System_data(u=norm['u_all'], y=norm['y_all'])
    sys_data_with_x.x = norm['x_all']

    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        Ad, Bd, Cd_dt, Dd_dt, sys_data_with_x)

    phys_encoder = linear_encoder_init(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        flag_linear_only=False,
    )

    encoder = LinearInitEncoderWrapper(
        phys_encoder=phys_encoder,
        nx_ann=HP['NX_ANN'],
        nb=nb + nb_right, nu=nu, na=na + na_right, ny=ny,
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        u_mean=u_mean, std_u=std_u, y0=y0, ystd=ystd,
        x_mean=x_mean, std_x=std_x,
    ).to(DTYPE_PT)

    return encoder


# =============================================================================
# Build full model at 400 Hz (intentional mismatch: 20 kHz encoder init)
# =============================================================================

def build_model_400hz(norm, encoder_20k, lambda_reg):
    """Build RegularizedSSE_Interconnect at 400 Hz.

    The model (RK4 integration, output block) runs at 400 Hz.
    The encoder is initialized from a deep copy of encoder_20k (20 kHz weights).
    A second frozen copy of encoder_20k is stored as the regularization reference.

    The encoder is set BEFORE init_model so the optimizer is built on the
    correct parameters (20 kHz init weights, not a fresh 400 Hz init).
    """
    ts = 1.0 / FS_TRAIN

    x_mean = norm['x_mean']
    std_x  = norm['std_x']
    std_u  = norm['std_u']
    u_mean = norm['u_mean']
    ystd   = norm['ystd']
    y0     = norm['y0']

    Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
    Dd_np   = Dd.numpy()
    PHY_IX  = np.arange(NX_PHYS)
    nxd     = NX_PHYS + HP['NX_ANN']

    ic = Interconnect(nxd, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=ts,
        up_sample=HP['up_sample'],
    ).to(DTYPE_PT)
    out_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    ic.add_block(phy_block)
    ic.add_block(out_block)

    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=nxd,
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=torch.nn.Tanh,
    )
    ic.add_block(ann_block)

    # Freeze ANN: NX_ANN=0 baseline, no mismatch correction
    for p in ann_block.parameters():
        p.requires_grad = False

    ic.connect_block_signals(ann_block, ["x", "u"], ["xp"])
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", out_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_block, ["u"], ["y"])

    # Frozen reference encoder: fixed copy of 20 kHz encoder, no gradients
    # This is the target the regularization pins x[:NX_PHYS] towards
    frozen_encoder = copy.deepcopy(encoder_20k)
    for p in frozen_encoder.parameters():
        p.requires_grad = False

    fit_sys = RegularizedSSE_Interconnect(
        interconnect=ic, na=na, nb=nb,
        na_right=na_right, nb_right=nb_right,
        frozen_encoder=frozen_encoder,
        lambda_reg=lambda_reg,
        nx_phys=NX_PHYS,
        e_net_kwargs={
            'n_nodes_per_layer': HP['n_nodes_per_layer'],
            'n_hidden_layers':   HP['n_hidden_layers'],
        },
    )

    fit_sys.norm.u0   = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0   = y0
    fit_sys.norm.ystd = ystd

    # Set encoder to 20 kHz init BEFORE init_model.
    # init_model calls init_nets which only creates an encoder if self.encoder is None.
    # By setting it here, the optimizer created in init_model picks up these weights.
    fit_sys.encoder = copy.deepcopy(encoder_20k)

    return fit_sys


# =============================================================================
# Direct encoder evaluation -- no simulation
# =============================================================================

def evaluate_encoder_direct(encoder, val_u, val_y, val_x, norm):
    """Evaluate encoder quality via direct sliding-window forward pass.

    For each timestep t >= _WIN_START, constructs the I/O window, calls the
    encoder, and compares to x_logical[t]. No model simulation is performed --
    this isolates encoder quality from model propagation.

    The norm argument determines how val_u and val_y are normalized before
    being fed to the encoder, and how the encoder output is un-normalized.
    When evaluating a 20 kHz encoder on 400 Hz data, norm_400 is passed:
    the slight statistics difference between rates is part of the mismatch
    being measured and is expected.

    Returns: (NX_PHYS,) per-state NRMS in physical units
    """
    u_mean = norm['u_mean'].flatten()
    std_u  = norm['std_u'].flatten()
    y0     = norm['y0']
    ystd   = norm['ystd']
    x_mean = norm['x_mean'].flatten()
    std_x  = norm['std_x'].flatten()

    u_norm = (val_u - u_mean) / std_u
    y_norm = (val_y - y0)    / ystd

    # Vectorized sliding windows -- no Python loop
    u_wins = np.lib.stride_tricks.sliding_window_view(
        u_norm, (_NB_WIN, nu)).reshape(-1, _NB_WIN, nu)
    y_wins = np.lib.stride_tricks.sliding_window_view(
        y_norm, (_NA_WIN, ny)).reshape(-1, _NA_WIN, ny)

    u_batch = torch.tensor(u_wins.copy(), dtype=DTYPE_PT)
    y_batch = torch.tensor(y_wins.copy(), dtype=DTYPE_PT)

    encoder.eval()
    with torch.no_grad():
        # Output is pipeline-normalized: (x_phys - x_mean) / std_x
        x_enc = encoder(u_batch, y_batch).numpy()

    x_phys = x_enc * std_x + x_mean   # physical units
    x_gt   = val_x[_WIN_START:]
    T      = min(len(x_phys), len(x_gt))
    x_phys = x_phys[:T]
    x_gt   = x_gt[:T]

    rms_err = np.sqrt(np.mean((x_phys - x_gt) ** 2, axis=0))
    rms_gt  = np.sqrt(np.mean(x_gt ** 2, axis=0))
    return rms_err / (rms_gt + 1e-12)


# =============================================================================
# Printing helpers
# =============================================================================

def print_nrms_header():
    h = f'  {"":>35s} |'
    for name in STATE_NAMES:
        h += f' {name:>10s} |'
    print(h, flush=True)
    print('  ' + '-' * (len(h) - 2), flush=True)


def print_nrms_row(label, nrms):
    row = f'  {label:>35s} |'
    for v in nrms:
        row += f' {v:10.3e} |'
    print(row, flush=True)


def print_epoch_header():
    h = f'{"epoch":>5s} | {"val_loss":>10s}'
    for name in STATE_NAMES:
        h += f' | {name:>9s}'
    print(h, flush=True)
    print('-' * (len(h) + len(STATE_NAMES) * 2), flush=True)


def print_epoch_row(epoch_label, loss_str, nrms, suffix=''):
    row = f'{epoch_label:>5s} | {loss_str:>10s}'
    for v in nrms:
        row += f' | {v:9.3e}'
    row += suffix
    print(row, flush=True)


# =============================================================================
# Main
# =============================================================================

def main():
    print('=' * 70, flush=True)
    print('Diagnostic: encoder mismatch + physical state regularization sweep', flush=True)
    print('=' * 70, flush=True)

    # =========================================================================
    # PART 1: Mismatch measurement
    # =========================================================================
    print('\n' + '=' * 70, flush=True)
    print('PART 1: Mismatch measurement (no training)', flush=True)
    print(f'  20 kHz encoder evaluated on 20 kHz data (reference)', flush=True)
    print(f'  Same encoder evaluated on {FS_TRAIN} Hz data (mismatch)', flush=True)
    print('=' * 70, flush=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- Load data at both rates ---
    d_orig  = 1                         # no downsampling for 20 kHz
    d_train = FS_ORIG // FS_TRAIN       # = 50 for 400 Hz

    print(f'\nLoading 20 kHz training data ({len(TRAIN_FILES)} files)...', flush=True)
    train_data_20k = [load_mat(f, d_orig) for f in TRAIN_FILES]
    val_u_20k, val_y_20k, val_x_20k = load_mat(VAL_FILE, d_orig)
    norm_20k = compute_normalization(train_data_20k)
    print(f'  Samples per file: {train_data_20k[0][0].shape[0]}', flush=True)

    print(f'\nLoading {FS_TRAIN} Hz training data ({len(TRAIN_FILES)} files)...', flush=True)
    train_data_400 = [load_mat(f, d_train) for f in TRAIN_FILES]
    val_u_400, val_y_400, val_x_400 = load_mat(VAL_FILE, d_train)
    norm_400 = compute_normalization(train_data_400)
    print(f'  Samples per file: {train_data_400[0][0].shape[0]}', flush=True)

    # --- Build 20 kHz encoder ---
    t0 = time.time()
    print(f'\nBuilding encoder at {FS_ORIG} Hz...', flush=True)
    encoder_20k = build_encoder_20khz(norm_20k)
    print(f'  Done ({time.time() - t0:.1f}s)', flush=True)

    # --- Evaluate ---
    nrms_20k_on_20k = evaluate_encoder_direct(
        encoder_20k, val_u_20k, val_y_20k, val_x_20k, norm_20k)
    nrms_20k_on_400 = evaluate_encoder_direct(
        encoder_20k, val_u_400, val_y_400, val_x_400, norm_400)
    ratio_mismatch = nrms_20k_on_400 / (nrms_20k_on_20k + 1e-12)

    # --- Print table ---
    print(flush=True)
    print_nrms_header()
    print_nrms_row(f'20kHz enc on 20kHz data (reference)', nrms_20k_on_20k)
    print_nrms_row(f'20kHz enc on {FS_TRAIN}Hz data (mismatch)', nrms_20k_on_400)
    print_nrms_row(f'ratio mismatch / reference', ratio_mismatch)

    worst_state = STATE_NAMES[int(np.argmax(ratio_mismatch))]
    print(f'\n  Worst mismatch ratio: {np.max(ratio_mismatch):.2f}x ({worst_state})', flush=True)
    print(f'  These are the epoch-0 starting values for Part 2.', flush=True)

    # --- Save Part 1 immediately ---
    part1_results = {
        'fs_encoder': FS_ORIG,
        'fs_train':   FS_TRAIN,
        'nrms_20k_on_20k': {n: float(nrms_20k_on_20k[i]) for i, n in enumerate(STATE_NAMES)},
        'nrms_20k_on_400': {n: float(nrms_20k_on_400[i]) for i, n in enumerate(STATE_NAMES)},
        'ratio':           {n: float(ratio_mismatch[i])   for i, n in enumerate(STATE_NAMES)},
    }
    part1_path = os.path.join(OUT_DIR, 'diagnostic_lambda_part1.json')
    with open(part1_path, 'w') as f:
        json.dump(part1_results, f, indent=2)
    print(f'\n  Saved: {part1_path}', flush=True)

    # =========================================================================
    # PART 2: Lambda sweep
    # =========================================================================
    print('\n' + '=' * 70, flush=True)
    print('PART 2: Lambda regularization sweep', flush=True)
    print(f'  Encoder init: 20 kHz weights (intentional mismatch)', flush=True)
    print(f'  Training:     {FS_TRAIN} Hz, nf={NF}, lr={LR:.0e}, epochs={N_EPOCHS}', flush=True)
    print(f'  Regularization: lambda * MSE(x_enc[:NX_PHYS], x_frozen[:NX_PHYS])', flush=True)
    print(f'  Lambda values: {LAMBDA_VALUES}', flush=True)
    print('=' * 70, flush=True)

    ts_train = 1.0 / FS_TRAIN
    train_list = [
        deepSI.System_data(u=u, y=y, dt=ts_train)
        for u, y, _ in train_data_400
    ]
    train_sys_data = deepSI.System_data_list(train_list)
    val_sys_data   = deepSI.System_data(u=val_u_400, y=val_y_400, dt=ts_train)

    results = {}

    for lambda_reg in LAMBDA_VALUES:
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        fit_sys = build_model_400hz(norm_400, encoder_20k, lambda_reg)
        fit_sys.init_model(
            sys_data=train_sys_data,
            auto_fit_norm=False,
            optimizer_kwargs={'lr': LR},
        )
        fit_sys.hfn.to(DTYPE_PT)

        print(f'\n{"=" * 100}', flush=True)
        print(f'lambda = {lambda_reg:.0e}', flush=True)
        print('-' * 100, flush=True)
        print_epoch_header()

        # Epoch 0: encoder init quality (direct eval, no training)
        nrms_0 = evaluate_encoder_direct(
            fit_sys.encoder, val_u_400, val_y_400, val_x_400, norm_400)
        state_curves = {n: [float(nrms_0[i])] for i, n in enumerate(STATE_NAMES)}
        print_epoch_row('init', '', nrms_0)

        # Train epoch by epoch, deepSI fit() epochs = cumulative total
        t0 = time.time()
        for epoch in range(1, N_EPOCHS + 1):
            fit_sys.fit(
                train_sys_data=train_sys_data,
                val_sys_data=val_sys_data,
                batch_size=HP['batch_size'],
                epochs=epoch,
                auto_fit_norm=False,
                loss_kwargs={'nf': NF},
                validation_measure=f'{NF}-step-RMS',
                verbose=False,
            )

            nrms_ep = evaluate_encoder_direct(
                fit_sys.encoder, val_u_400, val_y_400, val_x_400, norm_400)
            for i, name in enumerate(STATE_NAMES):
                state_curves[name].append(float(nrms_ep[i]))

            ep_loss = float(fit_sys.Loss_val[-1])
            print_epoch_row(str(epoch), f'{ep_loss:10.4e}', nrms_ep)

        elapsed = time.time() - t0
        loss_curve = [float(l) for l in fit_sys.Loss_val]

        # Per-state verdict vs init
        verdicts = {}
        for name in STATE_NAMES:
            init_v  = state_curves[name][0]
            final_v = state_curves[name][-1]
            r = final_v / init_v if init_v > 0 else float('inf')
            verdicts[name] = ('BETTER' if r < 0.8 else
                              'STABLE' if r < 1.2 else f'WORSE({r:.1f}x)')

        print(f'\n  Verdicts: ' +
              '  '.join(f'{n}:{v}' for n, v in verdicts.items()) +
              f'  ({elapsed:.1f}s)', flush=True)

        results[lambda_reg] = {
            'loss_curve':   loss_curve,
            'state_curves': state_curves,
            'verdicts':     verdicts,
            'elapsed_s':    elapsed,
        }

    # =========================================================================
    # Summary table
    # =========================================================================
    print('\n' + '=' * 70, flush=True)
    print('SUMMARY', flush=True)
    print('=' * 70, flush=True)

    # Print init -> final ratio per state for each lambda
    header = f'  {"lambda":>8s} | {"loss_0":>9s} -> {"loss_N":>9s}'
    for name in STATE_NAMES:
        header += f' | {name:>8s}'
    print(header, flush=True)
    print('  ' + '-' * (len(header) - 2), flush=True)

    for lambda_reg, res in results.items():
        lc  = res['loss_curve']
        row = f'  {lambda_reg:8.1e} | {lc[0]:9.3e} -> {lc[-1]:9.3e}'
        for name in STATE_NAMES:
            sc = res['state_curves'][name]
            r  = sc[-1] / sc[0] if sc[0] > 0 else float('inf')
            row += f' | {r:6.2f}x'
        print(row, flush=True)

    # =========================================================================
    # Save full JSON
    # =========================================================================
    json_out = {
        'config': {
            'fs_encoder': FS_ORIG,
            'fs_train':   FS_TRAIN,
            'nf': NF, 'lr': LR, 'n_epochs': N_EPOCHS,
            'lambda_values': LAMBDA_VALUES,
            'na': na, 'nb': nb, 'na_right': na_right, 'nb_right': nb_right,
            'train_files': TRAIN_FILES, 'val_file': VAL_FILE,
        },
        'part1': part1_results,
        'part2': {
            f'lambda={lv:.0e}': {
                'loss_curve':   res['loss_curve'],
                'state_curves': res['state_curves'],
                'verdicts':     res['verdicts'],
                'elapsed_s':    res['elapsed_s'],
            }
            for lv, res in results.items()
        },
    }
    json_path = os.path.join(OUT_DIR, 'diagnostic_lambda.json')
    with open(json_path, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f'\nSaved: {json_path}', flush=True)

    # =========================================================================
    # Plot 1: Per-state NRMS vs epoch, one line per lambda
    # Two reference lines per subplot:
    #   green dashed  = 20kHz encoder on 20kHz data (best achievable)
    #   orange dotted = 20kHz encoder on 400Hz data (mismatch start)
    # =========================================================================
    lambda_colors = plt.cm.plasma(np.linspace(0.0, 0.85, len(LAMBDA_VALUES)))

    fig, axes = plt.subplots(
        NX_PHYS, 1, figsize=(11, 3 * NX_PHYS), sharex=True)

    for si, name in enumerate(STATE_NAMES):
        ax = axes[si]
        for li, (lambda_reg, res) in enumerate(results.items()):
            sc = res['state_curves'][name]
            ax.plot(range(len(sc)), sc, 'o-',
                    color=lambda_colors[li], label=f'λ={lambda_reg:.0e}',
                    markersize=3, linewidth=1.2)

        # Reference: 20kHz encoder on 20kHz data (best possible)
        ax.axhline(y=float(nrms_20k_on_20k[si]), color='green',
                   linestyle='--', linewidth=1.0, alpha=0.7,
                   label='20kHz ref')
        # Reference: 20kHz encoder on 400Hz data (actual starting point)
        ax.axhline(y=float(nrms_20k_on_400[si]), color='orange',
                   linestyle=':', linewidth=1.0, alpha=0.7,
                   label='mismatch init')

        ax.set_ylabel(f'{name}\nNRMS')
        ax.set_yscale('log')
        ax.legend(fontsize=6, loc='upper right', ncol=4)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Epoch (0 = init)')
    fig.suptitle(
        f'Physical state NRMS vs epoch\n'
        f'(encoder: {FS_ORIG} Hz init, training: {FS_TRAIN} Hz, '
        f'nf={NF}, lr={LR:.0e})',
        fontsize=11,
    )
    fig.tight_layout()
    state_path = os.path.join(OUT_DIR, 'diagnostic_lambda_states.png')
    fig.savefig(state_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {state_path}', flush=True)

    # =========================================================================
    # Plot 2: Output loss vs epoch, one line per lambda
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 4))
    for li, (lambda_reg, res) in enumerate(results.items()):
        lc = res['loss_curve']
        ax.plot(range(1, len(lc) + 1), lc, 'o-',
                color=lambda_colors[li], label=f'λ={lambda_reg:.0e}',
                markersize=3, linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation loss')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f'Output validation loss vs epoch '
        f'(nf={NF}, lr={LR:.0e}, train: {FS_TRAIN} Hz)')
    fig.tight_layout()
    loss_path = os.path.join(OUT_DIR, 'diagnostic_lambda_loss.png')
    fig.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {loss_path}', flush=True)

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
