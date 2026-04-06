"""
train_param_recovery.py
-----------------------
Step 3b training script: recover true physical parameters from MATLAB data.

Approach: direct simulation (no encoder).
    x0 is known exactly ([0, 0, 0.3, 0, 0, 0] logical), so the deepSI
    encoder is unnecessary. We run simulate() directly and minimise:

        loss = MSE(Y_pred, q1_train) + block.param_loss()

    using Adam on block.log_params only. LFRFitSystem (Step 2) is kept for
    the future ANN augmentation experiment where x0 is genuinely unknown.

Why not SSE_Interconnect here:
    X1 and X2 output channels have std ~ 3e-7 m in this trajectory (the Y-sweep
    keeps X1=X2=0). auto_fit_norm divides by those stds, producing astronomically
    large normalised values that break the encoder and MSE computation.
    Direct simulation avoids the normalisation issue entirely.

Data: Matlab-output/lpv_sim_varying_y.mat
    True-parameter trajectory (ground truth, D-033).
    u_q1 (N,3) stage forces [N], q1 (N,3) stage positions [m], fs=20 kHz.

BPTT: truncated with segment_len=SEGMENT_LEN (fast, biased gradients).
    Parameters are constant across all steps so partial-trajectory gradients
    are informative and the truncation bias is acceptable.

Run as:
    conda run -n GraduationProject python -m lpv_lfr_baseline.train_param_recovery
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lpv_lfr_baseline.lfr_param_block import (
    ParameterizedLFRBlock, _build_matrices, _Lb, _d,
)
from lpv_lfr_baseline.lfr_simulate import simulate
from lpv_lfr_baseline.data_utils import compute_rmse_baseline, load_gantry_data
from lpv_lfr_baseline.physics import P, ts

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MAT_PATH   = os.path.join(os.path.dirname(__file__), '..', 'Matlab-output', 'lpv_sim_varying_y.mat')
SAVE_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models', 'gantry', 'param_recovery')

TRAIN_FRAC   = 0.8       # fraction of trajectory used for training
EPOCHS       = 500       # training epochs (increase for better convergence)
LR           = 1e-3      # Adam learning rate
SEGMENT_LEN  = 500       # truncated BPTT segment length (steps)
LOG_INTERVAL = 25        # print interval (epochs)

# Initial logical state: positions [0,0,0.3], velocities [0,0,0]
# Matches q1[0] = [0,0,0.3] in stage coords (see data_utils.py derivation)
X0_LOGICAL = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]], dtype=torch.float64)


def _load_tensors(mat_path, train_frac):
    """Load MATLAB data and split into train/val tensors."""
    mat    = loadmat(mat_path)
    u_all  = torch.tensor(mat['u_q1'], dtype=torch.float64)   # (N, 3)
    q1_all = torch.tensor(mat['q1'],   dtype=torch.float64)   # (N, 3)
    N      = u_all.shape[0]
    n_train = int(N * train_frac)

    u_train  = u_all[:n_train].unsqueeze(0)    # (1, n_train, 3)
    q1_train = q1_all[:n_train]                # (n_train, 3)
    u_val    = u_all[n_train:].unsqueeze(0)    # (1, n_val, 3)
    q1_val   = q1_all[n_train:]                # (n_val, 3)

    return u_train, q1_train, u_val, q1_val


def _simulate_no_grad(block, x0, u_seq):
    """Run a no-gradient simulation and return stage-coordinate output."""
    with torch.no_grad():
        params = torch.exp(block.log_params).clamp(min=1e-6)
        M0, M1, M2, K, C = _build_matrices(params, _Lb, _d)
        result = simulate(x0, u_seq, M0, M1, M2, K, C, P, ts, bptt_mode='full')
    return result.Y[0]   # (N, 3)


def train(epochs=EPOCHS, lr=LR, segment_len=SEGMENT_LEN,
          train_frac=TRAIN_FRAC, log_interval=LOG_INTERVAL,
          mat_path=MAT_PATH, save_dir=SAVE_DIR):
    """
    Run the parameter recovery training loop.

    Returns
    -------
    block : ParameterizedLFRBlock  -- trained block with recovered parameters
    """
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. RMSE_baseline (D-034) -- one forward pass, no gradient
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Computing RMSE_baseline from detuned baseline")
    print("=" * 60)
    rmse_baseline = compute_rmse_baseline(mat_path)
    print(f"  RMSE_baseline = {rmse_baseline:.6e} m  ({rmse_baseline*1e3:.4f} mm)")

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 2: Loading training and validation data")
    print("=" * 60)
    u_train, q1_train, u_val, q1_val = _load_tensors(mat_path, train_frac)
    print(f"  Train: {u_train.shape[1]} steps  ({u_train.shape[1]/20000:.3f} s)")
    print(f"  Val  : {u_val.shape[1]} steps  ({u_val.shape[1]/20000:.3f} s)")

    # ------------------------------------------------------------------
    # 3. Build block with computed RMSE_baseline
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3: Building ParameterizedLFRBlock")
    print("=" * 60)
    block = ParameterizedLFRBlock(RMSE_baseline=rmse_baseline)
    print(f"  RMSE_baseline used  : {rmse_baseline:.6e} m")
    print(f"  Trainable params    : {sum(p.numel() for p in block.parameters())}")
    print()
    print("Initial parameter table (detuned vs true):")
    print(block.param_table())

    # ------------------------------------------------------------------
    # 4. Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(block.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Step 4: Training  ({epochs} epochs, lr={lr}, segment_len={segment_len})")
    print("=" * 60)
    print(f"  {'Epoch':>6}  {'MSE [m^2]':>12}  {'param_loss':>12}  {'total':>12}  {'time [s]':>9}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*9}")

    best_val_mse = float('inf')
    t_start = time.time()

    for epoch in range(epochs):
        epoch_t0 = time.time()
        optimizer.zero_grad()

        # Forward: rebuild matrices from current log_params each epoch
        params = torch.exp(block.log_params).clamp(min=1e-6)
        M0, M1, M2, K, C = _build_matrices(params, _Lb, _d)

        result = simulate(
            X0_LOGICAL, u_train,
            M0, M1, M2, K, C, P, ts,
            bptt_mode='truncated', segment_len=segment_len,
        )
        y_pred = result.Y[0]   # (n_train, 3)

        mse_loss   = F.mse_loss(y_pred, q1_train)
        theta_loss = block.param_loss()
        total_loss = mse_loss + theta_loss

        total_loss.backward()
        optimizer.step()

        epoch_dt = time.time() - epoch_t0

        if epoch % log_interval == 0 or epoch == epochs - 1:
            print(f"  {epoch:>6}  {mse_loss.item():>12.4e}  "
                  f"{theta_loss.item():>12.4e}  "
                  f"{total_loss.item():>12.4e}  "
                  f"{epoch_dt:>9.3f}")

    total_time = time.time() - t_start
    print(f"\n  Training complete in {total_time:.1f} s  ({total_time/epochs:.2f} s/epoch)")

    # ------------------------------------------------------------------
    # 6. Validation MSE (no gradient)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 5: Validation")
    print("=" * 60)

    # Run detuned baseline (no training) on full trajectory for reference
    block_ref = ParameterizedLFRBlock(RMSE_baseline=rmse_baseline)
    y_pred_ref = _simulate_no_grad(block_ref, X0_LOGICAL, u_train)
    mse_ref = F.mse_loss(y_pred_ref, q1_train).item()

    # Run trained block on training data
    y_pred_train = _simulate_no_grad(block, X0_LOGICAL, u_train)
    mse_train = F.mse_loss(y_pred_train, q1_train).item()

    # Compute RMSE per channel
    err_ref   = (y_pred_ref   - q1_train).pow(2).mean(0).sqrt()
    err_train = (y_pred_train - q1_train).pow(2).mean(0).sqrt()
    ch = ['X1', 'X2', 'Y']
    print(f"  {'Channel':<6}  {'Detuned RMSE [mm]':>18}  {'Trained RMSE [mm]':>18}")
    print(f"  {'-'*6}  {'-'*18}  {'-'*18}")
    for i, name in enumerate(ch):
        print(f"  {name:<6}  {err_ref[i].item()*1e3:>18.4f}  {err_train[i].item()*1e3:>18.4f}")
    print(f"\n  Overall MSE: detuned={mse_ref:.4e}  trained={mse_train:.4e}")
    improved = mse_train < mse_ref
    print(f"  Prediction improved: {improved}")

    # ------------------------------------------------------------------
    # 7. Parameter recovery table
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 6: Parameter recovery")
    print("=" * 60)
    print(block.param_table())

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    save_path = os.path.join(save_dir, f'lfr_param_recovery_e{epochs}.pt')
    torch.save({
        'log_params':   block.log_params.detach(),
        'params_init':  block.params_init,
        'params_true':  block.params_init,   # placeholder -- true values in physics.py
        'RMSE_baseline': rmse_baseline,
        'epochs':        epochs,
        'lr':            lr,
        'train_mse':     mse_train,
    }, save_path)
    print(f"\nModel saved to: {save_path}")

    return block


if __name__ == '__main__':
    trained_block = train()
