"""
train_param_recovery.py
-----------------------
Step 3b: recover true physical parameters from MATLAB data using batched multiple shooting.

Approach:
    x0 is known exactly ([0, 0, 0.3, 0, 0, 0] logical), so no encoder is needed.
    Full state trajectory (positions + central-difference velocities) precomputed from
    data once and cached — parameter-free, independent of segmentation.
    Each epoch: n_seg segments sampled via stratified random indexing (one per stratum),
    guaranteeing full trajectory coverage every epoch. Segment start states from cached
    state_traj; u_seg and q1_seg built by vectorised advanced indexing each epoch.
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
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.physics import build_poly_constants
from lpv_lfr_baseline.core.lfr_simulate import simulate
from lpv_lfr_baseline.scripts.data_utils import compute_rmse_baseline

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MAT_PATH     = os.path.join(os.path.dirname(__file__), '..', '..', 'Matlab-output', 'lpv_sim_varying_y.mat')
SAVE_DIR     = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'gantry', 'param_recovery')

N_STEPS      = None  # cap on steps (None = use all); overridden to 500 when PROFILE=True
EPOCHS       = 1000     # training epochs
LR           = 1e-3  # Adam learning rate
SEGMENT_LEN       = 4000  # segment length - batch size = N_STEPS // SEGMENT_LEN
PARAM_LOSS_WEIGHT = 0.0   # 0.0 = disabled (parameter recovery), 1.0 = full (augmentation)
LOG_INTERVAL        = 25    # print every N epochs
CHECKPOINT_INTERVAL = 100   # save checkpoint_eN.pt every N epochs; 0 = disabled
PROFILE      = False  # profile epoch 0 and save report to SAVE_DIR/profile_out.txt
TIME_EPOCHS  = False   # print forward / backward timing each epoch

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
        params = block._recover_params()
        kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh = params
        M0, M1, M2, K, C = _build_matrices(
            torch.stack([kb1+kb2, cg1, cg2, cy, cb1+cb2, mh, m1, m2, mb, Jb+Jh]),
            block._Lb, block._d,
        )
        G = build_G_matrix(M0, M1, M2, K, C)
        alpha, beta, gamma, N0, N1, N2 = build_poly_constants(
            m1, m2, mb, mh, Jb, Jh, block._Lb, block._d
        )
        return simulate(
            x0, u, G, K, C, mh, alpha, beta, gamma, N0, N1, N2,
            block._P, block._ts, bptt_mode='full',
        )


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


_EPOCH_CACHE_SIZE = 10_000  # minimum epochs pre-generated in cache


def _get_epoch_indices(N_steps, n_seg, segment_len, epochs, device, save_dir):
    """
    Stratified random segment start indices for all training epochs.
    Trajectory [0, N_steps-segment_len] split into n_seg equal strata;
    one random start sampled per stratum per epoch — full coverage guaranteed every epoch.
    Cached independently of epoch count; reused as long as cache has >= epochs rows.
    """
    tag        = f'n{N_steps}_sl{segment_len}_nb{n_seg}'
    cache_path = os.path.join(save_dir, f'epoch_idx_{tag}.pt')
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu')
        if cached.shape[0] >= epochs:
            print(f'  epoch_idx: loaded from cache  ({tag})')
            return cached.to(device)
    n_gen   = max(_EPOCH_CACHE_SIZE, epochs)
    stratum = (N_steps - segment_len) // n_seg
    base    = torch.arange(n_seg).unsqueeze(0) * stratum            # (1, n_seg)
    offsets = torch.randint(0, stratum, (n_gen, n_seg))             # (n_gen, n_seg)
    idx     = (base + offsets).to(torch.int64)                      # (n_gen, n_seg)
    torch.save(idx.cpu(), cache_path)
    print(f'  epoch_idx: computed and cached  ({tag}, {n_gen} epochs)')
    return idx.to(device)


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
        params = self.block._recover_params()
        kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh = params
        params_10 = torch.stack([kb1+kb2, cg1, cg2, cy, cb1+cb2, mh, m1, m2, mb, Jb+Jh])
        M0, M1, M2, K, C = _build_matrices(params_10, self.block._Lb, self.block._d)
        G = build_G_matrix(M0, M1, M2, K, C)
        alpha, beta, gamma, N0, N1, N2 = build_poly_constants(
            m1, m2, mb, mh, Jb, Jh, self.block._Lb, self.block._d
        )
        return simulate(
            x0_seg, u_seg, G, K, C, mh, alpha, beta, gamma, N0, N1, N2,
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
    checkpoint_interval=CHECKPOINT_INTERVAL,
    mat_path=MAT_PATH, save_dir=SAVE_DIR, profile=PROFILE,
    time_epochs=TIME_EPOCHS, param_loss_weight=PARAM_LOSS_WEIGHT,
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
    print(f'  {N_steps} steps  →  {n_seg} × {segment_len} per epoch  '
          f'(stratified random, {N_steps - segment_len + 1} valid start positions)')

    # ------------------------------------------------------------------
    # 3. Block + optimizer
    # ------------------------------------------------------------------
    print(f'\n{"="*60}\nStep 3: Build model\n{"="*60}')
    block     = ParameterizedLFRBlock(RMSE_baseline=rmse_baseline).to(device)
    x0        = X0_LOGICAL.to(device)
    optimizer = torch.optim.Adam(block.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5,
    )
    n_gpus  = min(4, torch.cuda.device_count()) if device.type == 'cuda' else 0
    wrapper = torch.nn.DataParallel(_SimWrapper(block), device_ids=list(range(n_gpus))) \
              if n_gpus > 1 else _SimWrapper(block)
    print(f'  Trainable params : {sum(p.numel() for p in block.parameters())}')
    print(f'  RMSE_baseline    : {rmse_baseline:.6e} m')
    print(f'  GPUs in use      : {n_gpus if n_gpus > 1 else 1}  '
          f'({", ".join(torch.cuda.get_device_name(i) for i in range(max(n_gpus,1) if device.type == "cuda" else 0))})\n')
    print(block.param_table())

    # ------------------------------------------------------------------
    # 3b. Precomputed data structures - parameter-free
    # ------------------------------------------------------------------
    state_traj = _get_state_traj(q1_train, block._ts, device, save_dir)
    all_idx    = _get_epoch_indices(N_steps, n_seg, segment_len, epochs, device, save_dir)[:epochs]
    _arange_T  = torch.arange(segment_len, device=device)   # (T,) reused every epoch
    val_idx    = torch.arange(n_seg, device=device) * segment_len
    val_step   = val_idx.unsqueeze(1) + _arange_T
    val_x0     = state_traj[val_idx]                        # (n_seg, 6)  fixed every epoch
    val_u      = u_train[0][val_step]                       # (n_seg, T, 3)
    val_q1     = q1_train[val_step]                         # (n_seg, T, 3)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    print(f'\n{"="*60}\nStep 4: Train  ({epochs} epochs, lr={lr}, batch={n_seg}×{segment_len})\n{"="*60}')
    if param_loss_weight > 0:
        print(f'  {"Epoch":>6}  {"train_mse":>12}  {"param_loss":>12}  {"total":>12}  {"val_mse":>12}  {"grad_norm":>12}  {"time [s]":>9}')
        print(f'  {"-"*6}  {"-"*12}  {"-"*12}  {"-"*12}  {"-"*12}  {"-"*12}  {"-"*9}')
    else:
        print(f'  {"Epoch":>6}  {"train_mse":>12}  {"val_mse":>12}  {"grad_norm":>12}  {"time [s]":>9}')
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
        step_idx   = all_idx[epoch].unsqueeze(1) + _arange_T  # (n_seg, T)
        x0_seg     = state_traj[all_idx[epoch]]               # (n_seg, 6)
        u_seg      = u_train[0][step_idx]                     # (n_seg, T, 3)
        q1_seg     = q1_train[step_idx]                       # (n_seg, T, 3)

        with ctx as prof:
            Y_pred     = wrapper(x0_seg, u_seg)
            mse_loss   = F.mse_loss(Y_pred, q1_seg)
            theta_loss = block.param_loss() if param_loss_weight > 0 else None
            loss       = mse_loss + (param_loss_weight * theta_loss if theta_loss is not None else 0)
            t_fwd      = _sync_time(device)
            loss.backward()
            t_bwd      = _sync_time(device)

        if prof is not None:
            _save_profile(prof, save_dir)

        grad_norm = block.log_params.grad.norm().item() if block.log_params.grad is not None else float('nan')
        optimizer.step()

        if checkpoint_interval > 0 and epoch > 0 and epoch % checkpoint_interval == 0:
            torch.save({'log_params': block.log_params.detach(), 'epoch': epoch},
                       os.path.join(save_dir, f'checkpoint_e{epoch}.pt'))

        if epoch % log_interval == 0 or epoch == epochs - 1:
            with torch.no_grad():
                val_mse = F.mse_loss(wrapper(val_x0, val_u), val_q1).item()
            scheduler.step(val_mse)
            if param_loss_weight > 0:
                print(f'  {epoch:>6}  {mse_loss.item():>12.4e}  {theta_loss.item():>12.4e}  '
                      f'{loss.item():>12.4e}  {val_mse:>12.4e}  {grad_norm:>12.3e}  {time.time()-t0:>9.3f}', flush=True)
            else:
                print(f'  {epoch:>6}  {mse_loss.item():>12.4e}  {val_mse:>12.4e}  '
                      f'{grad_norm:>12.3e}  {time.time()-t0:>9.3f}', flush=True)
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
    save_path   = os.path.join(save_dir, f'lfr_param_recovery_e{epochs}_plw{param_loss_weight:.1f}.pt')
    torch.save({
        'log_params':         block.log_params.detach(),
        'params_init':        block.params_init,
        'params_true':        params_true,
        'RMSE_baseline':      rmse_baseline,
        'epochs':             epochs,
        'lr':                 lr,
        'segment_len':        segment_len,
        'param_loss_weight':  param_loss_weight,
        'train_mse':          mse_eval,
    }, save_path)
    print(f'\n  Saved to: {save_path}')

    return block


if __name__ == '__main__':
    train(profile=PROFILE, time_epochs=TIME_EPOCHS)

