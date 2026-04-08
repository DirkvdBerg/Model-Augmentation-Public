"""
train_param_recovery.py
-----------------------
Step 3b: recover true physical parameters from MATLAB data using batched multiple shooting.

Approach:
    x0 is known exactly ([0, 0, 0.3, 0, 0, 0] logical), so no encoder is needed.
    Each epoch:
      1. Pre-pass (no grad): simulate full trajectory to get n_seg segment start states.
      2. Training pass: simulate all n_seg segments in parallel (batch=n_seg, T=segment_len),
         full BPTT within each segment. Segment start states are detached - gradient flows
         only within each segment, matching the horizon of standard truncated BPTT.
    Loss: MSE(Y_pred, q1_train) + block.param_loss()
    Optimizer: Adam on block.log_params only.

Data: Matlab-output/lpv_sim_varying_y.mat
    True-parameter trajectory (ground truth, D-033).
    u_q1 (N,3) stage forces [N], q1 (N,3) stage positions [m], fs=20 kHz.

Run as:
    conda run -n GraduationProject python -m lpv_lfr_baseline.train_param_recovery
"""

import contextlib
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.profiler
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lpv_lfr_baseline.lfr_param_block import (
    ParameterizedLFRBlock, _build_matrices, _TRUE_PARAMS, _PARAM_NAMES,
)
from lpv_lfr_baseline.lfr_simulate import simulate as _simulate_eager
from lpv_lfr_baseline.data_utils import compute_rmse_baseline

_compile_backend = (
    'inductor'    # Triton-backed, fastest - requires CUDA capability >= 7.0
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    else 'aot_eager'  # fallback: works on CPU and older GPUs
)
simulate = torch.compile(_simulate_eager, backend=_compile_backend)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MAT_PATH     = os.path.join(os.path.dirname(__file__), '..', 'Matlab-output', 'lpv_sim_varying_y.mat')
SAVE_DIR     = os.path.join(os.path.dirname(__file__), '..', 'models', 'gantry', 'param_recovery')

N_STEPS      = None  # cap on steps (None = use all); overridden to 500 when PROFILE=True
EPOCHS       = 1     # training epochs
LR           = 1e-3  # Adam learning rate
SEGMENT_LEN  = 500   # segment length - batch size = N_STEPS // SEGMENT_LEN
LOG_INTERVAL = 25    # print every N epochs
PROFILE      = True  # profile epoch 0 and save report to SAVE_DIR/profile_out.txt

# Initial logical state: positions [0,0,0.3], velocities [0,0,0]
# Matches q1[0] = [0,0,0.3] in stage coords (see data_utils.py derivation)
X0_LOGICAL = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]], dtype=torch.float64)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _load_data(mat_path):
    """Load u_q1 and q1 from MATLAB file as float64 tensors."""
    mat = loadmat(mat_path)
    u   = torch.tensor(mat['u_q1'], dtype=torch.float64).unsqueeze(0)  # (1, N, 3)
    q1  = torch.tensor(mat['q1'],   dtype=torch.float64)               # (N, 3)
    return u, q1


def _run_no_grad(block, x0, u):
    """
    Simulate full trajectory with current block params, no gradient.
    Used for both the per-epoch pre-pass and the final evaluation.
    Returns SimResult with .X (1, N+1, 6) and .Y (1, N, 3).
    """
    with torch.no_grad():
        params = torch.exp(block.log_params).clamp(min=1e-6)
        M0, M1, M2, K, C = _build_matrices(params, block._Lb, block._d)
        return _simulate_eager(x0, u, M0, M1, M2, K, C, block._P, block._ts, bptt_mode='full')


def _save_profile(prof, save_dir):
    """Print profiler table to console and save to profile_out.txt."""
    table  = prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=20)
    header = '=' * 60 + '\nProfiler - epoch 0 (top 20 ops by self-CPU time)\n' + '=' * 60
    path   = os.path.join(save_dir, 'profile_out.txt')
    with open(path, 'w') as f:
        f.write(header + '\n' + table + '\n')
    print('\n' + header)
    print(table)
    print(f'  Saved to: {path}')


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(
    epochs=EPOCHS, lr=LR, segment_len=SEGMENT_LEN,
    n_steps=N_STEPS, log_interval=LOG_INTERVAL,
    mat_path=MAT_PATH, save_dir=SAVE_DIR, profile=PROFILE,
):
    """Run parameter recovery training. Returns trained ParameterizedLFRBlock."""
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'  Device: {torch.cuda.get_device_name(0)}  (CUDA {torch.version.cuda})')
    else:
        print('  Device: CPU')

    # ------------------------------------------------------------------
    # 1. RMSE_baseline (D-034)
    # ------------------------------------------------------------------
    print(f'\n{"="*60}\nStep 1: RMSE_baseline\n{"="*60}')
    rmse_baseline = compute_rmse_baseline(mat_path)
    print(f'  RMSE_baseline = {rmse_baseline:.6e} m  ({rmse_baseline*1e3:.4f} mm)')

    # ------------------------------------------------------------------
    # 2. Data
    # ------------------------------------------------------------------
    print(f'\n{"="*60}\nStep 2: Load and segment data\n{"="*60}')
    u_train, q1_train = _load_data(mat_path)
    if profile:
        n_steps = 500   # short run for profiling; bottleneck pattern is length-independent
    if n_steps is not None:
        u_train  = u_train[:, :n_steps, :]
        q1_train = q1_train[:n_steps]
    u_train  = u_train.to(device)
    q1_train = q1_train.to(device)

    N_steps = u_train.shape[1]
    n_seg   = N_steps // segment_len
    u_seg   = u_train[0, :n_seg * segment_len].reshape(n_seg, segment_len, 3)   # (n_seg, T, 3)
    q1_seg  = q1_train[:n_seg * segment_len].reshape(n_seg, segment_len, 3)     # (n_seg, T, 3)
    print(f'  {N_steps} steps  →  {n_seg} × {segment_len}  ({N_steps - n_seg*segment_len} steps dropped)')

    # ------------------------------------------------------------------
    # 3. Block + optimizer
    # ------------------------------------------------------------------
    print(f'\n{"="*60}\nStep 3: Build model\n{"="*60}')
    block     = ParameterizedLFRBlock(RMSE_baseline=rmse_baseline).to(device)
    x0        = X0_LOGICAL.to(device)
    optimizer = torch.optim.Adam(block.parameters(), lr=lr)
    print(f'  Trainable params : {sum(p.numel() for p in block.parameters())}')
    print(f'  RMSE_baseline    : {rmse_baseline:.6e} m\n')
    print(block.param_table())

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    print(f'\n{"="*60}\nStep 4: Train  ({epochs} epochs, lr={lr}, batch={n_seg}×{segment_len})\n{"="*60}')
    print(f'  {"Epoch":>6}  {"MSE [m²]":>12}  {"param_loss":>12}  {"total":>12}  {"time [s]":>9}')
    print(f'  {"-"*6}  {"-"*12}  {"-"*12}  {"-"*12}  {"-"*9}')

    pre     = None
    t_start = time.time()

    for epoch in range(epochs):
        t0 = time.time()
        optimizer.zero_grad()

        # Pre-pass: no-grad simulation to get accurate segment start states
        pre    = _run_no_grad(block, x0, u_train)
        x0_seg = pre.X[0, :n_seg * segment_len:segment_len, :]   # (n_seg, 6)

        # Training pass - profiled on epoch 0 when profile=True, no-op context otherwise
        ctx = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=False, with_stack=False,
            ) if (profile and epoch == 0) else contextlib.nullcontext()
        )
        with ctx as prof:
            params     = torch.exp(block.log_params).clamp(min=1e-6)
            M0, M1, M2, K, C = _build_matrices(params, block._Lb, block._d)
            result     = simulate(x0_seg, u_seg, M0, M1, M2, K, C, block._P, block._ts, bptt_mode='full')
            mse_loss   = F.mse_loss(result.Y, q1_seg)
            theta_loss = block.param_loss()
            loss       = mse_loss + theta_loss
            loss.backward()

        if prof is not None:
            _save_profile(prof, save_dir)

        optimizer.step()

        if epoch % log_interval == 0 or epoch == epochs - 1:
            print(f'  {epoch:>6}  {mse_loss.item():>12.4e}  {theta_loss.item():>12.4e}  '
                  f'{loss.item():>12.4e}  {time.time()-t0:>9.3f}', flush=True)

    if epochs > 1:
        total = time.time() - t_start
        print(f'\n  Done: {total:.1f} s  ({total/epochs:.2f} s/epoch)')

    # ------------------------------------------------------------------
    # 5. Evaluate - reuse last pre-pass, no extra simulation needed
    # ------------------------------------------------------------------
    print(f'\n{"="*60}\nStep 5: Prediction error\n{"="*60}')
    y_pred   = pre.Y[0]                                      # (N_steps, 3)
    mse_eval = F.mse_loss(y_pred, q1_train).item()
    err      = (y_pred - q1_train).pow(2).mean(0).sqrt()    # per-channel RMSE
    print(f'  {"Channel":<6}  {"RMSE [mm]":>12}')
    print(f'  {"-"*6}  {"-"*12}')
    for name, e in zip(['X1', 'X2', 'Y'], err):
        print(f'  {name:<6}  {e.item()*1e3:>12.4f}')
    print(f'\n  Overall MSE: {mse_eval:.4e} m²')

    # ------------------------------------------------------------------
    # 6. Parameter recovery table - primary go/no-go criterion
    # ------------------------------------------------------------------
    print(f'\n{"="*60}\nStep 6: Parameter recovery\n{"="*60}')
    print(block.param_table())

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    params_true = torch.tensor([_TRUE_PARAMS[n] for n in _PARAM_NAMES], dtype=torch.float64)
    save_path   = os.path.join(save_dir, f'lfr_param_recovery_e{epochs}.pt')
    torch.save({
        'log_params':    block.log_params.detach(),
        'params_init':   block.params_init,
        'params_true':   params_true,
        'RMSE_baseline': rmse_baseline,
        'epochs':        epochs,
        'lr':            lr,
        'train_mse':     mse_eval,
    }, save_path)
    print(f'\n  Saved to: {save_path}')

    return block


if __name__ == '__main__':
    train(profile=PROFILE)
