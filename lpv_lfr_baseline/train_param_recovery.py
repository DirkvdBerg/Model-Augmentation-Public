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
    Using direct simulation (known x0, no encoder needed) keeps the training
    loop minimal for parameter recovery. All three output channels (X1, X2, Y)
    have meaningful variation in lpv_sim_varying_y.mat, so auto_fit_norm would
    work fine -- switching to LFRFitSystem/SSE_Interconnect is viable if Jan's
    multiple-shooting batching is needed for speed.

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
import torch.profiler
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lpv_lfr_baseline.lfr_param_block import ParameterizedLFRBlock, _build_matrices
from lpv_lfr_baseline.lfr_simulate import simulate as _simulate_eager
from lpv_lfr_baseline.data_utils import compute_rmse_baseline

_compile_backend = (
    'inductor'   # Triton-backed, fastest - requires CUDA capability >= 7.0
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    else 'aot_eager'  # fallback: no Triton, works on CPU and old GPUs
)
simulate = torch.compile(_simulate_eager, backend=_compile_backend)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MAT_PATH   = os.path.join(os.path.dirname(__file__), '..', 'Matlab-output', 'lpv_sim_varying_y.mat')
SAVE_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models', 'gantry', 'param_recovery')

N_STEPS      = None      # cap on training steps per epoch (None = use all)
EPOCHS       = 1         # training epochs (increase for better convergence)
LR           = 1e-3      # Adam learning rate
SEGMENT_LEN  = 500       # segment length (steps): controls batch size = N // segment_len
LOG_INTERVAL = 25        # print interval (epochs)
PROFILE      = True     # set True to profile epoch 0 and print a time breakdown

# Initial logical state: positions [0,0,0.3], velocities [0,0,0]
# Matches q1[0] = [0,0,0.3] in stage coords (see data_utils.py derivation)
X0_LOGICAL = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]], dtype=torch.float64)


def _load_tensors(mat_path):
    """Load MATLAB data as tensors (no train/val split - param recovery uses full trajectory)."""
    mat    = loadmat(mat_path)
    u      = torch.tensor(mat['u_q1'], dtype=torch.float64).unsqueeze(0)  # (1, N, 3)
    q1     = torch.tensor(mat['q1'],   dtype=torch.float64)               # (N, 3)
    return u, q1


def _simulate_no_grad(block, x0, u_seq):
    """Run a no-gradient simulation and return stage-coordinate output."""
    with torch.no_grad():
        params = torch.exp(block.log_params).clamp(min=1e-6)
        M0, M1, M2, K, C = _build_matrices(params, block._Lb, block._d)
        result = simulate(x0, u_seq, M0, M1, M2, K, C, block._P, block._ts, bptt_mode='full')
    return result.Y[0]   # (N, 3)


def train(epochs=EPOCHS, lr=LR, segment_len=SEGMENT_LEN,
          n_steps=N_STEPS, log_interval=LOG_INTERVAL,
          mat_path=MAT_PATH, save_dir=SAVE_DIR, profile=PROFILE):
    """
    Run the parameter recovery training loop.

    Returns
    -------
    block : ParameterizedLFRBlock  -- trained block with recovered parameters
    """
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"  Device: {torch.cuda.get_device_name(0)}  (CUDA {torch.version.cuda})")
    else:
        print(f"  Device: CPU  (no CUDA GPU detected)")

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
    u_train, q1_train = _load_tensors(mat_path)
    # When profiling, cap at 500 steps so the profiler finishes quickly.
    # The bottleneck pattern is identical regardless of trajectory length.
    effective_n_steps = 500 if profile else n_steps
    if effective_n_steps is not None:
        u_train  = u_train[:, :effective_n_steps, :]
        q1_train = q1_train[:effective_n_steps]
    u_train  = u_train.to(device)
    q1_train = q1_train.to(device)
    N_steps  = u_train.shape[1]
    print(f"  Steps: {N_steps}  ({N_steps/20000:.3f} s)")

    # Pre-slice data into segments (constant across epochs — only x0_seg changes).
    # Last N_steps % segment_len steps are dropped (at most segment_len-1 steps lost).
    n_seg  = N_steps // segment_len
    u_seg  = u_train[0, :n_seg * segment_len, :].reshape(n_seg, segment_len, 3)  # (n_seg, T, 3)
    q1_seg = q1_train[:n_seg * segment_len, :].reshape(n_seg, segment_len, 3)    # (n_seg, T, 3)
    print(f"  Segments: {n_seg} × {segment_len} steps  ({N_steps - n_seg*segment_len} steps dropped)")

    # ------------------------------------------------------------------
    # 3. Build block with computed RMSE_baseline
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3: Building ParameterizedLFRBlock")
    print("=" * 60)
    block = ParameterizedLFRBlock(RMSE_baseline=rmse_baseline).to(device)
    x0 = X0_LOGICAL.to(device)
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
    print(f"Step 4: Training  ({epochs} epochs, lr={lr}, batch={n_seg}×{segment_len} steps)")
    print("=" * 60)
    print(f"  {'Epoch':>6}  {'MSE [m^2]':>12}  {'param_loss':>12}  {'total':>12}  {'time [s]':>9}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*9}")

    best_val_mse = float('inf')
    t_start = time.time()

    for epoch in range(epochs):
        epoch_t0 = time.time()
        print(f"  Epoch {epoch}/{epochs} running...", end='\r', flush=True)
        optimizer.zero_grad()

        # Pre-pass: get segment start states with current params (no gradient).
        # Uses _simulate_eager to avoid a separate compiled graph for the no-grad path.
        with torch.no_grad():
            params_pre = torch.exp(block.log_params).clamp(min=1e-6)
            M0_p, M1_p, M2_p, K_p, C_p = _build_matrices(params_pre, block._Lb, block._d)
            pre = _simulate_eager(
                x0, u_train,
                M0_p, M1_p, M2_p, K_p, C_p, block._P, block._ts,
                bptt_mode='full',
            )
        # pre.X: (1, N_steps+1, 6) — take state at start of each segment
        x0_seg = pre.X[0, :n_seg * segment_len:segment_len, :]  # (n_seg, 6)

        def _run_epoch():
            # Rebuild matrices with gradient for the training pass.
            params = torch.exp(block.log_params).clamp(min=1e-6)
            M0, M1, M2, K, C = _build_matrices(params, block._Lb, block._d)

            # Batched simulation: n_seg segments of segment_len steps in parallel.
            # x0_seg is detached (from no-grad pre-pass) so grad flows only within
            # each segment, not through the segment start states.
            result = simulate(
                x0_seg, u_seg,
                M0, M1, M2, K, C, block._P, block._ts,
                bptt_mode='full',
            )

            mse_loss   = F.mse_loss(result.Y, q1_seg)
            theta_loss = block.param_loss()
            total_loss = mse_loss + theta_loss
            total_loss.backward()
            return mse_loss, theta_loss, total_loss

        if profile and epoch == 0:
            # Profile epoch 0 only: records CPU time per op, then prints a
            # summary sorted by total self-CPU time (top 20 ops).
            # Disable CUDA activity recording -- float64 physics runs on CPU.
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=False,
                with_stack=False,
            ) as prof:
                mse_loss, theta_loss, total_loss = _run_epoch()
            prof_path = os.path.join(save_dir, 'profile_out.txt')
            os.makedirs(save_dir, exist_ok=True)
            table = prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=20
            )
            header = (
                "=" * 60 + "\n"
                "Profiler report - epoch 0 (top 20 ops by self-CPU time)\n"
                + "=" * 60
            )
            with open(prof_path, 'w') as f:
                f.write(header + "\n" + table + "\n")
            print()
            print(header)
            print(table)
            print(f"  Profiler report also saved to: {prof_path}")
        else:
            mse_loss, theta_loss, total_loss = _run_epoch()

        optimizer.step()

        epoch_dt = time.time() - epoch_t0

        if epoch % log_interval == 0 or epoch == epochs - 1:
            print(f"  {epoch:>6}  {mse_loss.item():>12.4e}  "
                  f"{theta_loss.item():>12.4e}  "
                  f"{total_loss.item():>12.4e}  "
                  f"{epoch_dt:>9.3f}", flush=True)

    total_time = time.time() - t_start
    if epochs > 1:
        print(f"\n  Training complete in {total_time:.1f} s  ({total_time/epochs:.2f} s/epoch)")

    # ------------------------------------------------------------------
    # 6. Simulation MSE on training data (no gradient)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 5: Prediction error on training data")
    print("=" * 60)

    y_pred_train = _simulate_no_grad(block, x0, u_train)
    mse_train    = F.mse_loss(y_pred_train, q1_train).item()
    err_train    = (y_pred_train - q1_train).pow(2).mean(0).sqrt()

    ch = ['X1', 'X2', 'Y']
    print(f"  {'Channel':<6}  {'Trained RMSE [mm]':>18}")
    print(f"  {'-'*6}  {'-'*18}")
    for i, name in enumerate(ch):
        print(f"  {name:<6}  {err_train[i].item()*1e3:>18.4f}")
    print(f"\n  Overall MSE: {mse_train:.4e} m²")

    # ------------------------------------------------------------------
    # 7. Parameter recovery table - primary go/no-go criterion
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 6: Parameter recovery (go/no-go criterion)")
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
    trained_block = train(profile=PROFILE)
