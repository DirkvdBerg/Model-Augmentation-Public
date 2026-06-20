"""
gantry_baseline_simulation.py
-----------------------------
Simulate the gantry FP model (Gantry_State_Block) on training/val/test data
and save the state trajectories. These are used for:

1. normalize_linear_ss_matrices() — needs state std to normalize A,B,C,D
2. Data-based encoder initialization (SS_pre_encoder, Eq. 35 in Hoekstra 2026)
3. Validation: comparing encoder-reconstructed states against FP model states

Usage:
    conda run -n GraduationProject python scripts/gantry/encoder_initialisation/gantry_baseline_simulation.py
"""

import os
import sys
import numpy as np
import torch
from scipy.io import loadmat

# --- Add project root to path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model_augmentation.fit_systems.blocks import Gantry_State_Block
from model_augmentation.systems.gantry_ss import P, Cd

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration (must match gantry_interconnect_dynamic.py)
# ═══════════════════════════════════════════════════════════════════════════════
NX_PHYS = 6
nu = 3
ny = 3
Y_OP = None  # LPV self-scheduled (same as training script)
UP_SAMPLE = 2

FS_ORIG = 20000
FS_NEW = 4000
D = FS_ORIG // FS_NEW
TS_NEW = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32

MODE = 'multisine'
DATA_SUBDIR = 'multisine' if MODE == 'multisine' else 'trajectories'
TRAJ_DIR = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab', DATA_SUBDIR)

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
TEST_FILE = 'E1_X_sym_anti_Y_low_offset_sweep.mat'

# Output directory
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'baseline_simulations',
                       f'{MODE}_LPV')
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Data loading (mirrors gantry_interconnect_dynamic.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_u(d):
    if 'u_total' in d:
        return d['u_total']
    return d['u']


def load_mat(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = _load_u(d)[::D].astype(DTYPE_NP)
    y = d['y'][::D].astype(DTYPE_NP)
    return u, y


# ═══════════════════════════════════════════════════════════════════════════════
# Compute normalization constants (same as training script)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_normalization(train_data_list):
    """Compute normalization constants from training data."""
    fs = FS_NEW
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)

    u_all = np.concatenate([u for u, _ in train_data_list])
    y_all = np.concatenate([y for _, y in train_data_list])

    x_logical_list = []
    for _, y in train_data_list:
        pos = (P_inv_T @ y.T).T
        vel = np.diff(pos, axis=0) * fs
        vel = np.vstack([vel[:1], vel])
        x_logical_list.append(np.hstack([pos, vel]))
    x_all = np.concatenate(x_logical_list)

    x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
    std_x = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
    std_u = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
    u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
    ystd = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    y0 = (Cd.numpy() @ x_mean.flatten()).astype(DTYPE_NP)

    return dict(x_mean=x_mean, std_x=std_x, std_u=std_u,
                u_mean=u_mean, ystd=ystd, y0=y0, P_inv_T=P_inv_T)


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline simulation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def simulate_baseline(u, y, norm, block):
    """
    Run the gantry FP model (Gantry_State_Block) on measured input u,
    collecting normalized state trajectories.

    Parameters
    ----------
    u : np.ndarray, (N, 3) — measured stage forces (raw, un-normalized)
    y : np.ndarray, (N, 3) — measured stage positions (raw, un-normalized)
    norm : dict — normalization constants
    block : Gantry_State_Block — the physics block

    Returns
    -------
    x_states : np.ndarray, (N, 6) — normalized state trajectory from FP model
    x_phys   : np.ndarray, (N, 6) — physical (un-normalized) state trajectory
    """
    N = u.shape[0]
    x_mean = norm['x_mean']  # (6,1)
    std_x = norm['std_x']    # (6,1)
    std_u = norm['std_u']    # (3,1)
    u_mean_val = norm['u_mean']    # (3,1)
    P_inv_T = norm['P_inv_T']

    # Initialize x0 from first measurement
    # THEORY: q_logical = P_inv @ y_stage (position from measurement equation)
    pos0 = P_inv_T @ y[0]  # (3,) logical positions
    vel0 = np.zeros(3, dtype=DTYPE_NP)  # assume zero initial velocity
    x0_phys = np.concatenate([pos0, vel0]).reshape(6, 1)  # (6,1)
    x0_norm = ((x0_phys - x_mean) / std_x).astype(DTYPE_NP)  # (6,1)

    # Normalize input sequence
    u_norm = ((u.T - u_mean_val) / std_u).T  # (N, 3) normalized

    # State storage
    x_states_norm = np.zeros((N, NX_PHYS), dtype=DTYPE_NP)
    x_states_phys = np.zeros((N, NX_PHYS), dtype=DTYPE_NP)

    # Initial state: block expects (batch, 6, 1)
    x_cur = torch.tensor(x0_norm, dtype=DTYPE_PT).unsqueeze(0)  # (1, 6, 1)

    x_states_norm[0] = x_cur.squeeze().numpy()
    x_states_phys[0] = x0_phys.flatten()

    for k in range(N - 1):
        u_k = torch.tensor(u_norm[k].reshape(1, nu, 1), dtype=DTYPE_PT)
        z = torch.cat([x_cur, u_k], dim=1)  # (1, 9, 1)
        x_next = block.nonlinear_function(z)  # (1, 6, 1)
        x_cur = x_next

        x_norm_np = x_next.squeeze().numpy()
        x_states_norm[k + 1] = x_norm_np
        x_states_phys[k + 1] = x_norm_np * std_x.flatten() + x_mean.flatten()

    return x_states_norm, x_states_phys


def main():
    print(f"Loading data from: {TRAJ_DIR}")
    print(f"Output dir: {OUT_DIR}")

    # Load training data
    train_data = [load_mat(f) for f in TRAIN_FILES]
    val_u, val_y = load_mat(VAL_FILE)
    test_u, test_y = load_mat(TEST_FILE)
    print(f"Loaded {len(train_data)} training trajectories, 1 val, 1 test")

    # Compute normalization
    norm = compute_normalization(train_data)
    print(f"Normalization computed: std_x = {norm['std_x'].flatten()}")

    # Build physics block
    block = Gantry_State_Block(
        Y_op=Y_OP, std_x=norm['std_x'], std_u=norm['std_u'],
        x_mean=norm['x_mean'], u_mean=norm['u_mean'],
        Ts=TS_NEW, up_sample=UP_SAMPLE,
    ).to(DTYPE_PT)
    block.eval()

    # Simulate on all datasets
    print("\nSimulating baseline on training trajectories...")
    x_train_norm_list = []
    x_train_phys_list = []
    for i, (fname, (u, y)) in enumerate(zip(TRAIN_FILES, train_data)):
        x_norm, x_phys = simulate_baseline(u, y, norm, block)
        x_train_norm_list.append(x_norm)
        x_train_phys_list.append(x_phys)
        print(f"  T{i+1} ({fname}): {x_norm.shape[0]} samples, "
              f"state RMS = {np.sqrt(np.mean(x_phys**2, axis=0)[:3])}")

    print("\nSimulating baseline on validation data...")
    x_val_norm, x_val_phys = simulate_baseline(val_u, val_y, norm, block)
    print(f"  Val: {x_val_norm.shape[0]} samples")

    print("\nSimulating baseline on test data...")
    x_test_norm, x_test_phys = simulate_baseline(test_u, test_y, norm, block)
    print(f"  Test: {x_test_norm.shape[0]} samples")

    # Save
    np.savez(
        os.path.join(OUT_DIR, 'baseline_states.npz'),
        # Normalized states (what the encoder should output)
        x_train_norm=[x for x in x_train_norm_list],
        x_val_norm=x_val_norm,
        x_test_norm=x_test_norm,
        # Physical states (for analysis / normalization computation)
        x_train_phys=[x for x in x_train_phys_list],
        x_val_phys=x_val_phys,
        x_test_phys=x_test_phys,
        # Normalization constants (for reproducibility)
        x_mean=norm['x_mean'],
        std_x=norm['std_x'],
        std_u=norm['std_u'],
        u_mean=norm['u_mean'],
        ystd=norm['ystd'],
        y0=norm['y0'],
    )
    print(f"\nSaved to {os.path.join(OUT_DIR, 'baseline_states.npz')}")

    # Quick sanity: compare simulated positions to measured positions
    print("\nSanity check: simulated vs measured output (first training trajectory)")
    u0, y0_meas = train_data[0]
    x0_phys = x_train_phys_list[0]
    Cd_np = Cd.numpy()
    y_sim = (Cd_np @ x0_phys.T).T  # (N, 3) simulated stage positions
    err = y0_meas - y_sim
    rms_per_ch = np.sqrt(np.mean(err**2, axis=0))
    print(f"  Per-channel output RMS error (X1, X2, Y): {rms_per_ch}")
    print(f"  Per-channel output std: {y0_meas.std(axis=0)}")
    nrms = rms_per_ch / (y0_meas.std(axis=0) + 1e-8) * 100
    print(f"  NRMS (%): {nrms}")


if __name__ == "__main__":
    main()
