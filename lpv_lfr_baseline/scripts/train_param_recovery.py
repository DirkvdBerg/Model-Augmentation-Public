"""
train_param_recovery.py
-----------------------
Step 3b: recover true physical parameters from MATLAB data using batched multiple shooting.

Approach:
    x0 is known exactly ([0, 0, 0.3, 0, 0, 0] logical), so no encoder is needed.
    Segment start states are computed once from data (parameter-free): positions
    read directly from q1_train, velocities from forward finite differences.
    Each epoch: simulate all n_seg segments in parallel (batch=n_seg, T=segment_len),
    full BPTT within each segment. Segment start states are fixed and detached.
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

from lpv_lfr_baseline.blocks.lfr_param_block import (
    ParameterizedLFRBlock, _build_matrices, _TRUE_PARAMS, _PARAM_NAMES,
)
from lpv_lfr_baseline.core.lfr_simulate import simulate
from lpv_lfr_baseline.scripts.data_utils import compute_rmse_baseline

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MAT_PATH     = os.path.join(os.path.dirname(__file__), '..', '..', 'Matlab-output', 'lpv_sim_varying_y.mat')
SAVE_DIR     = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'gantry', 'param_recovery')

N_STEPS      = None  # cap on steps (None = use all); overridden to 500 when PROFILE=True
EPOCHS       = 2     # training epochs
LR           = 1e-3  # Adam learning rate
SEGMENT_LEN  = 500   # segment length - batch size = N_STEPS // SEGMENT_LEN
LOG_INTERVAL = 25    # print every N epochs
PROFILE      = False  # profile epoch 0 and save report to SAVE_DIR/profile_out.txt
TIME_EPOCHS  = True   # print forward / backward timing each epoch

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
    """Simulate full trajectory with current params, no gradient. Used for evaluation."""
    with torch.no_grad():
        params = torch.exp(block.log_params).clamp(min=1e-6)
        M0, M1, M2, K, C = _build_matrices(params, block._Lb, block._d)
        return simulate(x0, u, M0, M1, M2, K, C, block._P, block._ts, bptt_mode='full')


def _get_state_traj(q1_train, ts, device, save_dir):
    """
    Full state trajectory from data — parameter-free.
    Positions from q1_train; velocities via central differences (O(ts²)),
    with forward/backward FD at the two boundary points.
    Result cached to disk by data length; independent of segmentation.
    """
    N          = q1_train.shape[0]
    tag        = f'n{N}'
    cache_path = os.path.join(save_dir, f'state_traj_{tag}.pt')
    if os.path.exists(cache_path):
        print(f'  state_traj: loaded from cache  ({tag})')
        return torch.load(cache_path, map_location=device)
    ts_val = float(ts)
    q      = q1_train.cpu()
    qdot   = torch.empty_like(q)
    qdot[0]    = (q[1]  - q[0])  / ts_val          # forward FD
    qdot[1:-1] = (q[2:] - q[:-2]) / (2 * ts_val)   # central differences
    qdot[-1]   = (q[-1] - q[-2]) / ts_val           # backward FD
    traj = torch.cat([q, qdot], dim=-1)         # (N, 6)
    torch.save(traj, cache_path)
    print(f'  state_traj: computed and cached  ({tag})')
    return traj.to(device)


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


class _SimWrapper(torch.nn.Module):
    """Thin wrapper so DataParallel can replicate block across GPUs.

    Keeps _build_matrices inside forward() so each GPU replica computes
    its own M0-C from its local copy of log_params instead of receiving
    pre-built matrices that DataParallel would incorrectly scatter along dim 0.
    """
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x0_seg, u_seg):
        params = torch.exp(self.block.log_params).clamp(min=1e-6)
        M0, M1, M2, K, C = _build_matrices(params, self.block._Lb, self.block._d)
        return simulate(
            x0_seg, u_seg, M0, M1, M2, K, C,
            self.block._P, self.block._ts, bptt_mode='full',
        ).Y


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def _sync_time(device):
    """Wall-clock time after synchronizing CUDA (accurate GPU timing)."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return time.time()


def train(
    epochs=EPOCHS, lr=LR, segment_len=SEGMENT_LEN,
    n_steps=N_STEPS, log_interval=LOG_INTERVAL,
    mat_path=MAT_PATH, save_dir=SAVE_DIR, profile=PROFILE,
    time_epochs=TIME_EPOCHS,
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
    n_gpus  = min(4, torch.cuda.device_count()) if device.type == 'cuda' else 0
    wrapper = torch.nn.DataParallel(_SimWrapper(block), device_ids=list(range(n_gpus))) \
              if n_gpus > 1 else _SimWrapper(block)
    print(f'  Trainable params : {sum(p.numel() for p in block.parameters())}')
    print(f'  RMSE_baseline    : {rmse_baseline:.6e} m')
    print(f'  GPUs in use      : {n_gpus if n_gpus > 1 else 1}  '
          f'({", ".join(torch.cuda.get_device_name(i) for i in range(max(n_gpus,1) if device.type == "cuda" else 0))})\n')
    print(block.param_table())

    # ------------------------------------------------------------------
    # 3b. Segment start states - precomputed from data, parameter-free
    # ------------------------------------------------------------------
    state_traj = _get_state_traj(q1_train, block._ts, device, save_dir)
    idx        = torch.arange(n_seg, device=device) * segment_len
    x0_seg     = state_traj[idx]   # (n_seg, 6)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    print(f'\n{"="*60}\nStep 4: Train  ({epochs} epochs, lr={lr}, batch={n_seg}×{segment_len})\n{"="*60}')
    print(f'  {"Epoch":>6}  {"MSE [m²]":>12}  {"param_loss":>12}  {"total":>12}  {"time [s]":>9}')
    print(f'  {"-"*6}  {"-"*12}  {"-"*12}  {"-"*12}  {"-"*9}')

    t_start = time.time()

    for epoch in range(epochs):
        t0 = _sync_time(device)
        optimizer.zero_grad()

        # Training pass - profiled on epoch 0 when profile=True, no-op context otherwise
        ctx = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=False, with_stack=False,
            ) if (profile and epoch == 0) else contextlib.nullcontext()
        )
        with ctx as prof:
            Y_pred     = wrapper(x0_seg, u_seg)
            mse_loss   = F.mse_loss(Y_pred, q1_seg)
            theta_loss = block.param_loss()
            loss       = mse_loss + theta_loss
            t_fwd      = _sync_time(device)
            loss.backward()
            t_bwd      = _sync_time(device)

        if prof is not None:
            _save_profile(prof, save_dir)

        optimizer.step()

        if epoch % log_interval == 0 or epoch == epochs - 1:
            print(f'  {epoch:>6}  {mse_loss.item():>12.4e}  {theta_loss.item():>12.4e}  '
                  f'{loss.item():>12.4e}  {time.time()-t0:>9.3f}', flush=True)
        if time_epochs:
            print(f'    fwd={t_fwd-t0:.2f}s  bwd={t_bwd-t_fwd:.2f}s  '
                  f'total={t_bwd-t0:.2f}s', flush=True)

    if epochs > 1:
        total = time.time() - t_start
        print(f'\n  Done: {total:.1f} s  ({total/epochs:.2f} s/epoch)')

    # ------------------------------------------------------------------
    # 5. Evaluate - fresh post-training pre-pass (pre may be stale)
    # ------------------------------------------------------------------
    print(f'\n{"="*60}\nStep 5: Prediction error\n{"="*60}')
    pre    = _run_no_grad(block, x0, u_train)
    y_pred = pre.Y[0]                                        # (N_steps, 3)
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
    train(profile=PROFILE, time_epochs=TIME_EPOCHS)

