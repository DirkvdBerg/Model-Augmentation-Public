"""
train_param_recovery.py
-----------------------
Parameter recovery training for the dual-gantry LPV-LFR baseline.

Approach
--------
1. precompute() loads/builds all fixed data (trajectories, sigma, segment_len) once.
2. Each epoch:
   a. torch.compiler.cudagraph_mark_step_begin() — signal new step to CUDA graph backend.
   b. Build G, K, C and polynomial constants once from current block.log_params.
   c. Sample a uniform segment batch.
   d. Windowed BPTT (window size W): simulate W steps per window, accumulate MSE
      losses into a single tensor, detach state between windows. Single backward.
   e. split_loss() backward separately (independent graph).
   f. optimizer.step(); optimizer.zero_grad(set_to_none=True).
3. Full-trajectory eval every VALIDATION_INTERVAL epochs; best params tracked.
4. Post-training: restore best params, final eval, param table, save.

No _SimWrapper, no DataParallel. Direct simulate() calls throughout.
All channels contribute to the loss — sigma (from precompute) normalizes each channel.

Run as:
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.train_param_recovery
"""

import contextlib
import os
import queue
import threading
import time

import torch

from lpv_lfr_baseline.blocks.lfr_param_block import (
    ParameterizedLFRBlock,
    _PARAM_NAMES,
    _TRUE_PARAMS,
    _DETUNED_PARAMS,
    _build_matrices,
)
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.lfr_simulate import simulate
from lpv_lfr_baseline.core.physics import build_poly_constants
from lpv_lfr_baseline.scripts.precompute import precompute, _build_segment_pools, load_eval_trajs

# ── Dtype (single toggle — flows into precompute and all .to() calls) ────────
DTYPE = torch.float64

# ── Dataset selector ─────────────────────────────────────────────────────────
# Options:
#   'base'          — T1–T6, pure trajectory, no multisine
#                     Matlab-output/parameter-recovery/
#   'multisine'     — T1–T8, multisine injected as plant-input force (old approach)
#                     Matlab-output/parameter-recovery-multisine/
#   'ref_injection' — T1–T8, multisine injected into reference r (preferred)
#                     Matlab-output/parameter-recovery-ref-injection/
#                     Excitation reaches plant via T≈1, not attenuated by S≪1.
DATASET = 'ref_injection'

_BASE = os.path.join(os.path.dirname(__file__), '..', '..')
_TRAJ_BASE = (
    {'id': 'T1', 'file': 'T1_Y_sweep_conservative.mat'},
    {'id': 'T2', 'file': 'T2_X_sym_Y030.mat'},
    {'id': 'T3', 'file': 'T3_X_sym_Y000.mat'},
    {'id': 'T4', 'file': 'T4_X_antisym_Y020.mat'},
    {'id': 'T5', 'file': 'T5_X_sym_Y_sweep.mat'},
    {'id': 'T6', 'file': 'T6_Y_sweep_aggressive.mat'},
)
_TRAJ_EXTENDED = _TRAJ_BASE + (
    {'id': 'T7', 'file': 'T7_X_antisym_Y_sweep.mat'},
    {'id': 'T8', 'file': 'T8_X_sym_anti_Y_sweep.mat'},
)
_DATASETS = {
    'base': dict(
        traj_dir   = os.path.join(_BASE, 'Matlab-output', 'parameter-recovery'),
        save_dir   = os.path.join(_BASE, 'simulations', 'param_recovery'),
        traj_specs = _TRAJ_BASE,
    ),
    'multisine': dict(
        traj_dir   = os.path.join(_BASE, 'Matlab-output', 'parameter-recovery-multisine'),
        save_dir   = os.path.join(_BASE, 'simulations', 'param_recovery_multisine'),
        traj_specs = _TRAJ_EXTENDED,
    ),
    'ref_injection': dict(
        traj_dir   = os.path.join(_BASE, 'Matlab-output', 'parameter-recovery-ref-injection'),
        save_dir   = os.path.join(_BASE, 'simulations', 'param_recovery_ref_injection'),
        traj_specs = _TRAJ_EXTENDED,
    ),
}

_ds        = _DATASETS[DATASET]
TRAJ_DIR   = _ds['traj_dir']
SAVE_DIR   = _ds['save_dir']
TRAJ_SPECS = _ds['traj_specs']

# ── Validation / test trajectories ───────────────────────────────────────────
# Separate directory; files not used during training.
_VAL_TEST_DIR = os.path.join(_BASE, 'Matlab-output', 'parameter-recovery-ref-injection-val-test')

# VAL_SPECS: drives LR scheduler and best-model selection during training.
# () = fall back to training trajectories (no held-out validation).
VAL_SPECS = ({'id': 'V1', 'file': 'V1_X_sym_Y_mid_sweep.mat'},)

# TEST_SPECS: evaluated in Step 4 only; never seen during training.
# () = skip test evaluation.
TEST_SPECS = ({'id': 'E1', 'file': 'E1_X_sym_anti_Y_low_offset_sweep.mat'},)

# ── Normalisation ────────────────────────────────────────────────────────────
NORM_MODE = 'global'   # 'per_traj' | 'global'  (see precompute.py)

# ── Segment sampling ──────────────────────────────────────────────────────────
OVERLAP_FRACTION = 0.0   # 0.0 = non-overlapping; 0.5 = 50% overlap

# ── Training hyperparameters ──────────────────────────────────────────────────
W                  = 50      # BPTT window [samples] — outer loop in train()
EPOCHS             = 600
LR                 = 1e-3
VALIDATION_INTERVAL = None     # Set to None when not used. The lr scheduler steps on validation RMSE every VALIDATION_INTERVAL epochs; best params tracked.
LOG_INTERVAL             = 1
CHECKPOINT_INTERVAL      = 100
SPLIT_REG_WEIGHT         = 1e-2
N_STEPS                  = None    # cap on trajectory steps (None = all); set to 500 when PROFILE=True
PROFILE                  = False
TIME_EPOCHS              = False
BASE_SEED                = 1234

# Defensive: cudagraph_mark_step_begin is a PyTorch 2.x API — guard for older builds.
_MARK_STEP_BEGIN = getattr(torch.compiler, 'cudagraph_mark_step_begin', None)


# ── Segment pool helpers ──────────────────────────────────────────────────────

def _traj_set_tag(specs):
    return '_'.join(s['id'] for s in specs)


def _sample_batch(trajs, pools, segment_len, seed):
    """
    Sample one segment uniformly at random from each trajectory.
    Batch order matches trajs order.

    Returns
    -------
    x0_batch  : (B, 6)     initial states
    u_batch   : (B, S, 3)  stage-force inputs
    q1_batch  : (B, S, 3)  stage positions (targets)
    """
    rng = torch.Generator().manual_seed(seed)
    x0_list, u_list, q1_list = [], [], []
    for traj in trajs:
        starts = pools[traj['id']]
        idx    = int(torch.randint(len(starts), (1,), generator=rng))
        s      = starts[idx]
        x0_list.append(traj['state_traj'][s])
        u_list.append(traj['u'][0, s:s + segment_len])
        q1_list.append(traj['q1'][s:s + segment_len])
    return (
        torch.stack(x0_list),    # (B, 6)
        torch.stack(u_list),     # (B, S, 3)
        torch.stack(q1_list),    # (B, S, 3)
    )


# ── Physics helpers ───────────────────────────────────────────────────────────

def _build_sim_params(block):
    """
    Build all matrices needed for simulate() from current block parameters.

    Returns G, K, C, mh, alpha, beta, gamma, N0, N1, N2 — all differentiable
    w.r.t. block.log_params so gradients flow back to the parameters.
    """
    params = block._recover_params()
    kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d = params
    params_10 = torch.stack([kb1 + kb2, cg1, cg2, cy, cb1 + cb2, mh, m1, m2, mb, Jb + Jh])
    _, M1, M2, K, C = _build_matrices(params_10, block._Lb, d)
    alpha, beta, gamma, N0, N1, N2 = build_poly_constants(
        m1, m2, mb, mh, Jb, Jh, block._Lb, d
    )
    d0 = mh * (alpha * gamma - beta ** 2)
    G  = build_G_matrix(N0, d0, M1, M2, K, C)
    return G, K, C, mh, alpha, beta, gamma, N0, N1, N2


@torch._dynamo.disable
def _run_no_grad(block, x0, u, ts_tensor):
    """Simulate full trajectory with current params, no gradient.

    Decorated with @torch._dynamo.disable (recursive) so compiled functions
    called within (rk4_step) run eagerly. This is required for two reasons:
    1. The eval worker thread does not have the TLS state that cudagraphs needs.
    2. Eval runs infrequently so compilation overhead is not worth it.
    """
    with torch.no_grad():
        G, K, C, mh, alpha, beta, gamma, N0, N1, N2 = _build_sim_params(block)
        return simulate(
            x0, u, G, K, C, mh, alpha, beta, gamma, N0, N1, N2,
            block._P, ts_tensor, bptt_mode='full', return_latents=False,
        )


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _aggregate_rmse(entries):
    """Simple mean RMSE across all trajectories."""
    return (sum(e['mse_total'] for e in entries) / len(entries)) ** 0.5


def _full_traj_eval(block, trajs, ts_tensor):
    """Full-trajectory eval (no grad). Returns (mean_rmse_m, entries)."""
    entries = []
    for traj in trajs:
        result   = _run_no_grad(block, traj['state_traj'][:1], traj['u'], ts_tensor)
        diff     = result.Y[0] - traj['q1']
        rmse_ch  = diff.pow(2).mean(dim=0).sqrt().cpu()
        rmse_tot = float(diff.pow(2).mean().item() ** 0.5)
        entries.append({
            'id':         traj['id'],
            'mse_total':  rmse_tot ** 2,
            'rmse_total': rmse_tot,
            'rmse_ch':    rmse_ch,
        })
    return _aggregate_rmse(entries), entries


def _vram_fits(traj_list, dtype, dev, headroom=0.8):
    """Return True if all trajectory tensors fit in free VRAM (with headroom).

    Always returns True on CPU (no limit). headroom=0.8 reserves 20% for
    model weights, optimizer states, and intermediate activations.
    """
    if dev.type != 'cuda':
        return True
    itemsize = torch.finfo(dtype).bits // 8
    needed   = sum(
        (t['u'].numel() + t['q1'].numel() + t['state_traj'].numel()) * itemsize
        for t in traj_list
    )
    free, _  = torch.cuda.mem_get_info(dev)
    fits     = needed < free * headroom
    needed_mb = needed / 2**20
    free_mb   = free   / 2**20
    return fits, needed_mb, free_mb


# ── Logging helpers ───────────────────────────────────────────────────────────

def _sync_time(device):
    """Wall-clock time after CUDA synchronization (accurate GPU timing)."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return time.time()


def _save_profile(prof, save_dir):
    """Print profiler table to console and save text + Chrome trace."""
    table  = prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=20)
    header = '=' * 60 + '\nProfiler — top 20 ops by self-CPU time\n' + '=' * 60
    path   = os.path.join(save_dir, 'profile_out.txt')
    with open(path, 'w') as f:
        f.write(header + '\n' + table + '\n')
    print('\n' + header)
    print(table)
    print(f'  Saved text log to: {path}')
    trace_path = os.path.join(save_dir, 'profile_trace.json')
    print('  Exporting Chrome trace (may take a few minutes)...', flush=True)
    prof.export_chrome_trace(trace_path)
    print(f'  Saved Chrome trace to: {trace_path}')


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    epochs=EPOCHS,
    lr=LR,
    n_steps=N_STEPS,
    log_interval=LOG_INTERVAL,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    save_dir=SAVE_DIR,
    profile=PROFILE,
    time_epochs=TIME_EPOCHS,
    split_reg_weight=SPLIT_REG_WEIGHT,
    norm_mode=NORM_MODE,
):
    """Run parameter recovery training. Returns trained ParameterizedLFRBlock."""
    os.makedirs(save_dir, exist_ok=True)
    traj_tag = _traj_set_tag(TRAJ_SPECS)   # e.g. 'T1_T2_T3_T4_T5_T6_T7_T8'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'  Device: {torch.cuda.get_device_name(0)}  (CUDA {torch.version.cuda})')
    else:
        print('  Device: CPU')

    # ------------------------------------------------------------------
    # Step 1 — Precompute (cache-backed)
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}\nStep 1: Precompute (trajectories, sigma, segment_len)\n{"=" * 60}')
    pre = precompute(TRAJ_SPECS, TRAJ_DIR, save_dir, dtype=DTYPE, norm_mode=norm_mode,
                     overlap_fraction=OVERLAP_FRACTION)

    trajs                    = pre['trajs']
    sigma                    = pre['sigma']               # dict traj_id -> (3,) CPU float64
    rmse_baseline_normalized = pre['rmse_baseline_normalized']
    segment_len              = pre['segment_len']
    pools                    = pre['pools']               # dict traj_id -> list[int] start indices
    ts_eff                   = float(pre['ts_eff'])       # effective timestep after decimation [s]
    fs_new                   = float(pre['fs_new'])       # sampling rate after decimation [Hz]
    D                        = int(pre['D'])              # decimation factor

    # Move trajectory tensors to training device — pre-flight VRAM check.
    # Fast path: all tensors resident on GPU (zero per-epoch H2D transfer).
    # Slow path: tensors stay on CPU; _sample_batch slices on CPU, then the
    #            small segment batch (~300 KB) is moved to device each epoch.
    fits, needed_mb, free_mb = _vram_fits(trajs, DTYPE, device)
    trajs_on_gpu = fits
    if trajs_on_gpu:
        print(f'  Trajectory preload: GPU  ({needed_mb:.0f} MB needed, {free_mb:.0f} MB free)')
        for traj in trajs:
            traj['u']          = traj['u'].to(device=device, dtype=DTYPE)
            traj['q1']         = traj['q1'].to(device=device, dtype=DTYPE)
            traj['state_traj'] = traj['state_traj'].to(device=device, dtype=DTYPE)
    else:
        print(f'  Trajectory preload: CPU  ({needed_mb:.0f} MB needed, {free_mb:.0f} MB free — per-epoch H2D transfer)')
        for traj in trajs:
            traj['u']          = traj['u'].to(dtype=DTYPE)
            traj['q1']         = traj['q1'].to(dtype=DTYPE)
            traj['state_traj'] = traj['state_traj'].to(dtype=DTYPE)

    # Pre-build device-side sigma lookup — avoids per-epoch .to() inside the loop
    sigma_device = {tid: s.to(device=device, dtype=DTYPE) for tid, s in sigma.items()}

    # Effective timestep tensor — replaces block._ts in all simulate() calls
    ts_tensor = torch.tensor(ts_eff, dtype=DTYPE, device=device)

    # Eval device: cuda:1 when available (separate compile cache, true GPU parallelism);
    # falls back to training device (sync eval, no thread).
    eval_device = torch.device('cuda:1') if torch.cuda.device_count() >= 2 else device
    async_eval  = eval_device != device
    if async_eval:
        print(f'  Eval device: {torch.cuda.get_device_name(1)} (cuda:1, async)')
    else:
        print(f'  Eval device: same as training (sync, no thread)')

    # Validation trajectories: held-out if VAL_SPECS is set, else reuse training trajs.
    # Moved to eval_device so the async worker (2-GPU) can use them directly.
    val_trajs_raw = load_eval_trajs(VAL_SPECS, _VAL_TEST_DIR, D, dtype=DTYPE) if VAL_SPECS else trajs
    val_fits, val_needed_mb, val_free_mb = _vram_fits(val_trajs_raw, DTYPE, eval_device)
    val_dev   = eval_device if val_fits else torch.device('cpu')
    val_trajs = [{**_t, 'u': _t['u'].to(device=val_dev, dtype=DTYPE),
                  'q1': _t['q1'].to(device=val_dev, dtype=DTYPE),
                  'state_traj': _t['state_traj'].to(device=val_dev, dtype=DTYPE)}
                 for _t in val_trajs_raw]
    print(f'  Val preload: {"GPU" if val_fits else "CPU"}  ({val_needed_mb:.0f} MB needed, {val_free_mb:.0f} MB free)')

    # Test trajectories: evaluated in Step 4 only; kept on CPU to avoid consuming VRAM during training.
    test_trajs = load_eval_trajs(TEST_SPECS, _VAL_TEST_DIR, D, dtype=DTYPE) if TEST_SPECS else []

    print(f'  rmse_baseline_normalized = {rmse_baseline_normalized:.4e}')
    print(f'  segment_len = {segment_len} samples')

    if profile:
        n_steps = 500
    if n_steps is not None:
        for traj in trajs:
            traj['u']          = traj['u'][:, :n_steps, :]
            traj['q1']         = traj['q1'][:n_steps]
            traj['state_traj'] = traj['state_traj'][:n_steps]
            traj['N']          = min(traj['N'], n_steps)
        segment_len = min(segment_len, n_steps)
        pools = _build_segment_pools(trajs, segment_len, OVERLAP_FRACTION)  # truncation changes valid starts
    n_windows = (segment_len + W - 1) // W

    # sigma_batch is constant each epoch: one entry per trajectory in trajs order
    sigma_batch = torch.stack([sigma_device[traj['id']] for traj in trajs])  # (B, 3)

    pool_summary = {tid: len(s) for tid, s in pools.items()}
    stride = max(1, round(segment_len * (1 - OVERLAP_FRACTION)))
    print(f'  Active: {", ".join(s["id"] for s in TRAJ_SPECS)}')
    print(
        f'  segment_len={segment_len}, W={W}, n_windows={n_windows}, '
        f'batch={len(trajs)} trajs/epoch'
    )
    print(
        f'  fs_new={fs_new:.0f} Hz (D={D}), overlap={OVERLAP_FRACTION:.0%}, stride={stride}'
    )
    print(f'  Pool sizes: {pool_summary}')

    # ------------------------------------------------------------------
    # Step 2 — Block + optimizer
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}\nStep 2: Build model\n{"=" * 60}')
    block     = ParameterizedLFRBlock(RMSE_baseline=rmse_baseline_normalized).to(
        device=device, dtype=DTYPE
    )
    optimizer = torch.optim.Adam(block.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5, min_lr=1e-5,
    )
    print(f'  Trainable params : {sum(p.numel() for p in block.parameters())}')
    print(f'  RMSE_baseline_normalized: {rmse_baseline_normalized:.6e}')
    print(block.param_table())

    # ------------------------------------------------------------------
    # Step 3 — Training loop
    # ------------------------------------------------------------------
    print(
        f'\n{"=" * 60}\nStep 3: Train  '
        f'({epochs} epochs, lr={lr}, W={W})\n{"=" * 60}'
    )
    _eval_suffix = f'  |  {"eval_ep":>7}  {"eval_rmse":>12}' if VALIDATION_INTERVAL is not None else ''
    print(f'  {"Epoch":>6}  {"mse_loss":>12}  {"split_reg":>12}  {"grad_norm":>10}  {"time [s]":>9}  {"lr":>10}' + _eval_suffix)
    _sep_suffix  = f'  |  {"-" * 7}  {"-" * 12}' if VALIDATION_INTERVAL is not None else ''
    print(f'  {"-" * 6}  {"-" * 12}  {"-" * 12}  {"-" * 10}  {"-" * 9}  {"-" * 10}' + _sep_suffix)

    t_start             = time.time()
    history             = []
    latest_eval_epoch   = '-'
    latest_eval_rmse    = '-'
    best_full_traj_rmse = float('inf')
    best_epoch          = -1
    best_log_params     = None

    # Async eval infrastructure: only created when eval_device != device (2-GPU case).
    # On 1 GPU: eval runs inline in the training loop (no thread, no compile-cache race).
    if async_eval:
        eval_block   = ParameterizedLFRBlock(RMSE_baseline=rmse_baseline_normalized).to(
            device=eval_device, dtype=DTYPE
        )
        ts_eval_tensor = torch.tensor(ts_eff, dtype=DTYPE, device=eval_device)
        snap_queue   = queue.Queue(maxsize=2)
        result_queue = queue.Queue()

        def _eval_worker():
            while True:
                item = snap_queue.get()
                if item is None:
                    snap_queue.task_done()
                    break
                snap_epoch, lp_cpu = item
                with torch.no_grad():
                    eval_block.log_params.copy_(
                        lp_cpu.to(device=eval_device, dtype=eval_block.log_params.dtype)
                    )
                rmse, entries = _full_traj_eval(eval_block, val_trajs, ts_eval_tensor)
                result_queue.put((snap_epoch, rmse, entries, lp_cpu))
                snap_queue.task_done()

        threading.Thread(target=_eval_worker, daemon=True).start()

    prof = None
    if profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        prof = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
        )
        prof.start()

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        with (torch.profiler.record_function(f'Epoch {epoch}') if profile
              else contextlib.nullcontext()):

            t0 = _sync_time(device)

            # Signal new step to CUDA graph backend (no-op on CPU or older PyTorch)
            if _MARK_STEP_BEGIN is not None:
                _MARK_STEP_BEGIN()

            # Build G once per epoch from current parameters — used for all windows
            G, K, C, mh, alpha, beta, gamma, N0, N1, N2 = _build_sim_params(block)

            # Sample one segment per trajectory — guaranteed coverage each epoch
            x0_seg, u_seg, q1_seg = _sample_batch(
                trajs, pools, segment_len, BASE_SEED + 10_000 + epoch
            )
            if not trajs_on_gpu:
                x0_seg = x0_seg.to(device)
                u_seg  = u_seg.to(device)
                q1_seg = q1_seg.to(device)

            # Windowed BPTT — accumulate into a single loss tensor, then one backward.
            # State is detached between windows (truncated BPTT); G is shared across all
            # windows so gradients flow back through the shared prefix log_params -> G.
            x_win    = x0_seg    # (B, 6)
            mse_loss = None      # accumulated as a computation-graph tensor

            for w in range(n_windows):
                w_start = w * W
                w_end   = min(w_start + W, segment_len)
                u_win   = u_seg[:, w_start:w_end, :]     # (B, w_len, 3)
                q1_win  = q1_seg[:, w_start:w_end, :]    # (B, w_len, 3)

                result  = simulate(
                    x_win, u_win,
                    G, K, C, mh, alpha, beta, gamma, N0, N1, N2,
                    block._P, ts_tensor,
                    bptt_mode='full', return_latents=False,
                )

                # Normalize by sigma (all channels contribute, no masking)
                err      = (result.Y - q1_win) / sigma_batch.unsqueeze(1)  # (B, w_len, 3)
                win_loss = err.pow(2).mean() / n_windows   # scale so sum = mean over windows
                mse_loss = win_loss if mse_loss is None else mse_loss + win_loss

                x_win = result.X[:, -1, :].detach()   # carry state, stop gradient

            mse_loss.backward()

            # split_reg backward separately — independent graph, accumulates to log_params.grad
            split_reg_val = 0.0
            if split_reg_weight > 0:
                split_reg     = block.split_loss() * split_reg_weight
                split_reg.backward()
                split_reg_val = split_reg.item()

            t_fwd = _sync_time(device)   # fwd + bwd complete

        grad_norm = (
            block.log_params.grad.norm().item()
            if block.log_params.grad is not None
            else float('nan')
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if checkpoint_interval > 0 and epoch > 0 and epoch % checkpoint_interval == 0:
            torch.save(
                {'log_params': block.log_params.detach(), 'epoch': epoch, 'history': history},
                os.path.join(save_dir, f'checkpoint_e{epoch}.pt'),
            )

        # ── Eval (async drain+push or sync inline) ────────────────────────────
        _eval_rmse_this_epoch = None
        if async_eval:
            while not result_queue.empty():
                snap_epoch, full_rmse, _, lp_cpu = result_queue.get_nowait()
                latest_eval_epoch = snap_epoch
                latest_eval_rmse  = f'{full_rmse:.4e}'
                scheduler.step(full_rmse)
                for h in history:
                    if h['epoch'] == snap_epoch:
                        h['full_traj_rmse_m'] = full_rmse
                        h['log_params_snapshot'] = lp_cpu
                        break
                if full_rmse < best_full_traj_rmse:
                    best_full_traj_rmse = full_rmse
                    best_epoch          = snap_epoch
                    best_log_params     = lp_cpu

            if VALIDATION_INTERVAL is not None and (epoch % VALIDATION_INTERVAL == 0 or epoch == epochs - 1):
                try:
                    snap = block.log_params.detach().cpu().clone()
                    if epoch == epochs - 1:
                        snap_queue.put((epoch, snap))
                    else:
                        snap_queue.put_nowait((epoch, snap))
                except queue.Full:
                    pass

        else:
            if VALIDATION_INTERVAL is not None and (epoch % VALIDATION_INTERVAL == 0 or epoch == epochs - 1):
                lp_cpu    = block.log_params.detach().cpu().clone()
                full_rmse, _ = _full_traj_eval(block, val_trajs, ts_tensor)
                latest_eval_epoch = epoch
                latest_eval_rmse  = f'{full_rmse:.4e}'
                scheduler.step(full_rmse)
                if full_rmse < best_full_traj_rmse:
                    best_full_traj_rmse = full_rmse
                    best_epoch          = epoch
                    best_log_params     = lp_cpu
                _eval_rmse_this_epoch = full_rmse

        if epoch % log_interval == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            elapsed    = time.time() - t0
            _row_suffix = (f'  |  {latest_eval_epoch!s:>7}  {latest_eval_rmse!s:>12}'
                           if VALIDATION_INTERVAL is not None else '')
            print(
                f'  {epoch:>6}  {mse_loss.item():>12.4e}  {split_reg_val:>12.4e}  '
                f'{grad_norm:>10.3e}  {elapsed:>9.3f}  {current_lr:>10.3e}' + _row_suffix,
                flush=True,
            )
            entry = {
                'epoch':      epoch,
                'mse_loss':   mse_loss.item(),
                'split_reg':  split_reg_val,
                'grad_norm':  grad_norm,
                'lr':         current_lr,
            }
            if _eval_rmse_this_epoch is not None:
                entry['full_traj_rmse_m']    = _eval_rmse_this_epoch
                entry['log_params_snapshot'] = block.log_params.detach().cpu().clone()
            history.append(entry)

        if time_epochs:
            print(f'    fwd+bwd={t_fwd - t0:.2f}s', flush=True)

    # Finalize profiler
    if prof is not None:
        print('\n  Aggregating profiler trace...', flush=True)
        prof.stop()
        _save_profile(prof, save_dir)

    # Drain any remaining async eval results
    if async_eval:
        snap_queue.put(None)
        snap_queue.join()
        while not result_queue.empty():
            snap_epoch, full_rmse, _, lp_cpu = result_queue.get_nowait()
            for h in history:
                if h['epoch'] == snap_epoch:
                    h['full_traj_rmse_m'] = full_rmse
                    h['log_params_snapshot'] = lp_cpu
                    break
            if full_rmse < best_full_traj_rmse:
                best_full_traj_rmse = full_rmse
                best_epoch          = snap_epoch
                best_log_params     = lp_cpu

    if epochs > 1:
        total = time.time() - t_start
        print(f'\n  Done: {total:.1f} s  ({total / epochs:.2f} s/epoch)')

    # ------------------------------------------------------------------
    # Step 4 — Restore best params and final eval
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}\nStep 4: Final evaluation (best epoch = {best_epoch})\n{"=" * 60}')

    if best_log_params is not None:
        with torch.no_grad():
            block.log_params.copy_(
                best_log_params.to(device=device, dtype=block.log_params.dtype)
            )
        print(
            f'  Loaded best_log_params from epoch {best_epoch} '
            f'(full-traj RMSE = {best_full_traj_rmse:.4e} m)\n'
        )

    _hdr = f'  {"Traj":<6}  {"RMSE [m]":>12}  {"X1 [m]":>10}  {"X2 [m]":>10}  {"Y [m]":>10}'
    _sep = f'  {"-" * 6}  {"-" * 12}  {"-" * 10}  {"-" * 10}  {"-" * 10}'

    def _eval_group(traj_list):
        entries = []
        for traj in traj_list:
            _x0 = traj['state_traj'][:1].to(device)
            _u  = traj['u'].to(device)
            _q1 = traj['q1'].to(device)
            result   = _run_no_grad(block, _x0, _u, ts_tensor)
            diff     = result.Y[0] - _q1
            rmse_ch  = diff.pow(2).mean(dim=0).sqrt().cpu()
            rmse_tot = float(diff.pow(2).mean().item() ** 0.5)
            entries.append({'id': traj['id'], 'mse_total': rmse_tot ** 2,
                            'rmse_total': rmse_tot, 'rmse_ch': rmse_ch})
            print(f'  {traj["id"]:<6}  {rmse_tot:>12.4e}'
                  f'  {rmse_ch[0]:>10.4e}  {rmse_ch[1]:>10.4e}  {rmse_ch[2]:>10.4e}')
        return entries

    print('\n  [Training]')
    print(_hdr); print(_sep)
    train_eval_entries = _eval_group(trajs)
    train_overall_rmse = _aggregate_rmse(train_eval_entries)
    print(f'\n  Train RMSE: {train_overall_rmse:.6e} m')

    val_eval_entries = []
    val_overall_rmse = None
    if VAL_SPECS:
        print('\n  [Validation]')
        print(_hdr); print(_sep)
        val_eval_entries = _eval_group(val_trajs)
        val_overall_rmse = _aggregate_rmse(val_eval_entries)
        print(f'\n  Val RMSE:   {val_overall_rmse:.6e} m')

    test_eval_entries = []
    test_overall_rmse = None
    if TEST_SPECS:
        print('\n  [Test]')
        print(_hdr); print(_sep)
        test_eval_entries = _eval_group(test_trajs)
        test_overall_rmse = _aggregate_rmse(test_eval_entries)
        print(f'\n  Test RMSE:  {test_overall_rmse:.6e} m')

    # ------------------------------------------------------------------
    # Step 5 — Parameter recovery table
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}\nStep 5: Parameter recovery\n{"=" * 60}')
    print(block.param_table())

    # ------------------------------------------------------------------
    # Step 6 — Save
    # ------------------------------------------------------------------
    params_true    = torch.tensor([_TRUE_PARAMS[n] for n in _PARAM_NAMES], dtype=DTYPE)
    params_learned = block.params_init * block.log_params.detach().exp()

    # Identifiable sums
    _sum_pairs = [('kb_sum', 'kb1', 'kb2'), ('cb_sum', 'cb1', 'cb2'), ('J_sum', 'Jb', 'Jh')]
    _idx       = {n: i for i, n in enumerate(_PARAM_NAMES)}
    sum_names         = [sn for sn, _, _ in _sum_pairs]
    sum_params_true   = torch.tensor([_TRUE_PARAMS[a]    + _TRUE_PARAMS[b]    for _, a, b in _sum_pairs], dtype=DTYPE)
    sum_params_init   = torch.tensor([_DETUNED_PARAMS[a] + _DETUNED_PARAMS[b] for _, a, b in _sum_pairs], dtype=DTYPE)
    sum_params_learned = torch.stack([params_learned[_idx[a]] + params_learned[_idx[b]] for _, a, b in _sum_pairs])
    sum_delta_pct      = (sum_params_learned - sum_params_true) / sum_params_true * 100

    train_eval_rmse_ch = (
        sum(e['rmse_ch'].pow(2) for e in train_eval_entries) / len(train_eval_entries)
    ).sqrt()   # (3,) mean per-channel RMSE across training trajectories

    save_path = os.path.join(save_dir, f'lfr_param_recovery_{DATASET}_{traj_tag}_e{epochs}.pt')
    torch.save(
        {
            # Parameters — individual
            'param_names':              list(_PARAM_NAMES),
            'params_true':              params_true,
            'params_init':              block.params_init,
            'params_learned':           params_learned,
            'log_params':               block.log_params.detach(),
            # Parameters — identifiable sums
            'sum_names':                sum_names,
            'sum_params_true':          sum_params_true,
            'sum_params_init':          sum_params_init,
            'sum_params_learned':       sum_params_learned,
            'sum_delta_pct':            sum_delta_pct,
            # Best-epoch tracking
            'best_epoch':               best_epoch,
            'best_full_traj_rmse':      best_full_traj_rmse,
            'best_log_params':          best_log_params,
            # Normalisation
            'rmse_baseline_normalized': rmse_baseline_normalized,
            'sigma':                    {tid: s.cpu() for tid, s in sigma.items()},
            # Run config
            'active_traj_ids':          tuple(s['id'] for s in TRAJ_SPECS),
            'dtype':                    str(DTYPE),
            'norm_mode':                norm_mode,
            'epochs':                   epochs,
            'lr':                       lr,
            'segment_len':              segment_len,
            'overlap_fraction':         OVERLAP_FRACTION,
            'W':                        W,
            'split_reg_weight':         split_reg_weight,
            'batch_size':               len(trajs),
            'base_seed':                BASE_SEED,
            # Results — training trajectories
            'eval_train_rmse':          train_overall_rmse,
            'eval_train_rmse_ch':       train_eval_rmse_ch,
            'eval_train_entries':       train_eval_entries,
            # Results — validation trajectory
            'eval_val_rmse':            val_overall_rmse,
            'eval_val_entries':         val_eval_entries,
            # Results — test trajectory
            'eval_test_rmse':           test_overall_rmse,
            'eval_test_entries':        test_eval_entries,
            'history':                  history,
        },
        save_path,
    )
    print(f'\n  Saved to: {save_path}')

    return block


if __name__ == '__main__':
    train(profile=PROFILE, time_epochs=TIME_EPOCHS)
