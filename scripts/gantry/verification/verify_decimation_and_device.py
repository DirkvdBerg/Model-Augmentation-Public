"""
verify_decimation_and_device.py
-------------------------------
Tests that decimation (20kHz → 1kHz) preserves physics accuracy, and benchmarks
CPU vs GPU speed for the gantry Interconnect.

Part 1: Standalone block rollout at 20kHz vs 1kHz — physics error from decimation
Part 2: Full Interconnect single step at both rates
Part 3: Normalization stats at 20kHz vs 1kHz
Part 4: CPU vs GPU speed benchmark (forward steps + batch fwd/bwd)
Part 5: CPU vs GPU numerical consistency

Run: conda run -n GraduationProject python scripts/gantry/verification/verify_decimation_and_device.py
"""

import os
import sys
import time
import numpy as np
import torch
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model_augmentation.utils.utils import selection_matrix, expansion_matrix
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect
from model_augmentation.fit_systems.blocks import (
    Gantry_State_Block, Linear_Output_Block, Static_ANN_Block,
)
from model_augmentation.systems.gantry_ss import Cd, Dd

# ── Config ────────────────────────────────────────────────────────────────────
NX_PHYS   = 6
NX_ANN    = 2
nxd       = NX_PHYS + NX_ANN
nu, ny    = 3, 3
Y_OP      = None         # LPV
PHY_IX    = np.arange(NX_PHYS)

FS_ORIG   = 20000
D         = 20           # decimation factor
FS_NEW    = FS_ORIG // D
TS_ORIG   = 1.0 / FS_ORIG
TS_NEW    = 1.0 / FS_NEW
N_STEPS   = 100          # steps at 1kHz (= 100ms); equivalent to D*N_STEPS=2000 at 20kHz

BENCH_STEPS   = 200      # forward steps for speed benchmark
BENCH_WARMUP  = 10       # warmup iterations (excluded from timing)
BATCH_SIZE    = 256
NF            = 200

HAS_CUDA = torch.cuda.is_available()

# ── Load one trajectory ──────────────────────────────────────────────────────
traj_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'Matlab-output', 'identification-trajectories-no-multisine')
d = loadmat(os.path.join(traj_dir, 'T1_Y_sweep_conservative.mat'), squeeze_me=True)

u_20k = d['u_total'].astype(np.float32)   # (N, 3) at 20kHz
y_20k = d['q1'].astype(np.float32)        # (N, 3) at 20kHz
u_1k  = u_20k[::D]
y_1k  = y_20k[::D]

print(f'Loaded T1: {u_20k.shape[0]} samples at 20kHz, {u_1k.shape[0]} samples at 1kHz')

# ── Helper: compute normalization stats ──────────────────────────────────────
def compute_norm_stats(u, y, fs):
    vel = np.diff(y, axis=0) * fs
    vel = np.vstack([vel[:1], vel])
    x_logical = np.hstack([y, vel])
    return {
        'x_mean': x_logical.mean(axis=0),
        'std_x':  x_logical.std(axis=0) + 1e-8,
        'std_u':  u.std(axis=0) + 1e-8,
        'u_mean': u.mean(axis=0),
        'ystd':   y.std(axis=0) + 1e-8,
    }

stats_20k = compute_norm_stats(u_20k, y_20k, FS_ORIG)
stats_1k  = compute_norm_stats(u_1k,  y_1k,  FS_NEW)

# Use 1kHz stats for all block builds (consistent with training script)
x_mean = stats_1k['x_mean'].reshape(NX_PHYS, 1).astype(np.float32)
std_x  = stats_1k['std_x'].reshape(NX_PHYS, 1).astype(np.float32)
std_u  = stats_1k['std_u'].reshape(nu, 1).astype(np.float32)
u_mean = stats_1k['u_mean'].reshape(nu, 1).astype(np.float32)
ystd   = stats_1k['ystd'].astype(np.float32)
y0     = (Cd.numpy() @ x_mean.flatten()).astype(np.float32)
Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
Dd_np   = Dd.numpy()


# ── Helper: build interconnect ───────────────────────────────────────────────
def build_interconnect(ts):
    ic = Interconnect(nxd, nu, ny, debugging=False)
    sb = Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u,
                             x_mean=x_mean, u_mean=u_mean, Ts=ts)
    ob = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    ab = Static_ANN_Block(nz=nxd+nu, nw=nxd, n_nodes_per_layer=64,
                           net=zero_init_feed_forward_nn, activation=torch.nn.Tanh)
    ic.add_block(sb)
    ic.add_block(ob)
    ic.add_block(ab)
    ic.connect_block_signals(ab, ["x", "u"], ["xp"])
    ic.connect_signals("x", sb, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(sb, ["u"], [])
    ic.connect_signals(sb, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", ob, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(ob, ["u"], ["y"])
    return ic


def rollout(ic, x0_phys, u_raw, n_steps):
    """Run n_steps through interconnect, return physical y."""
    x_norm_0 = (x0_phys - x_mean.flatten()) / std_x.flatten()
    u_norm   = (u_raw[:n_steps] - u_mean.flatten()) / std_u.flatten()

    x = torch.zeros(1, nxd, dtype=torch.float32)
    x[0, :NX_PHYS] = torch.tensor(x_norm_0, dtype=torch.float32)
    u_t = torch.tensor(u_norm, dtype=torch.float32)

    y_out = np.zeros((n_steps, ny), dtype=np.float32)
    with torch.no_grad():
        for t in range(n_steps):
            y_t, x = ic.forward(x, u_t[t:t+1])
            y_out[t] = y_t.squeeze().numpy() * ystd + y0
    return y_out


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Decimation physics error — standalone block rollout
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'Part 1: Standalone block — 20kHz vs 1kHz ({N_STEPS} steps at 1kHz)')
print(f'{"="*70}')

# 20kHz block: run D*N_STEPS steps
block_20k = Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u,
                                x_mean=x_mean, u_mean=u_mean, Ts=TS_ORIG)
# 1kHz block: run N_STEPS steps
block_1k  = Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u,
                                x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW)

# Initial state: first sample of y_20k as positions, zero velocities
x0_phys = np.zeros(NX_PHYS, dtype=np.float32)
x0_phys[:3] = y_20k[0]
x_norm_0 = (x0_phys - x_mean.flatten()) / std_x.flatten()

# 20kHz rollout
u_norm_20k = (u_20k[:D*N_STEPS] - u_mean.flatten()) / std_u.flatten()
x_20k = torch.tensor(x_norm_0.reshape(1, NX_PHYS, 1), dtype=torch.float32)
y_20k_rollout = []
with torch.no_grad():
    for t in range(D * N_STEPS):
        u_t = torch.tensor(u_norm_20k[t].reshape(1, nu, 1), dtype=torch.float32)
        z = torch.cat([x_20k, u_t], dim=1)
        x_20k = block_20k.nonlinear_function(z)
        if (t + 1) % D == 0:
            x_phys = (x_20k.squeeze() * std_x.flatten() + x_mean.flatten()).numpy()
            y_20k_rollout.append(Cd.numpy() @ x_phys)
y_20k_rollout = np.array(y_20k_rollout)  # (N_STEPS, 3)

# 1kHz rollout
u_norm_1k = (u_1k[:N_STEPS] - u_mean.flatten()) / std_u.flatten()
x_1k = torch.tensor(x_norm_0.reshape(1, NX_PHYS, 1), dtype=torch.float32)
y_1k_rollout = []
with torch.no_grad():
    for t in range(N_STEPS):
        u_t = torch.tensor(u_norm_1k[t].reshape(1, nu, 1), dtype=torch.float32)
        z = torch.cat([x_1k, u_t], dim=1)
        x_1k = block_1k.nonlinear_function(z)
        x_phys = (x_1k.squeeze() * std_x.flatten() + x_mean.flatten()).numpy()
        y_1k_rollout.append(Cd.numpy() @ x_phys)
y_1k_rollout = np.array(y_1k_rollout)  # (N_STEPS, 3)

err_1 = np.abs(y_1k_rollout - y_20k_rollout)
nrms_1 = np.sqrt(((y_1k_rollout - y_20k_rollout)**2).mean(axis=0)) / ystd
print(f'  Ts_orig={TS_ORIG:.1e}  Ts_new={TS_NEW:.1e}  D={D}')
print(f'  {N_STEPS}-step rollout max|y_1kHz - y_20kHz| per channel:')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    print(f'    {lbl}: max={err_1[:, ch].max():.3e}  NRMS={nrms_1[ch]:.3e}')


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Full Interconnect — single step at both rates
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'Part 2: Interconnect single-step — 20kHz vs 1kHz')
print(f'{"="*70}')

ic_20k = build_interconnect(TS_ORIG)
ic_1k  = build_interconnect(TS_NEW)

# Single step at 20kHz: run D steps, take last output
x_ic_20k = torch.zeros(1, nxd, dtype=torch.float32)
x_ic_20k[0, :NX_PHYS] = torch.tensor(x_norm_0, dtype=torch.float32)
u_norm_20k_ic = (u_20k[:D] - u_mean.flatten()) / std_u.flatten()
with torch.no_grad():
    for t in range(D):
        y_t_20k, x_ic_20k = ic_20k.forward(
            x_ic_20k, torch.tensor(u_norm_20k_ic[t:t+1], dtype=torch.float32))
y_phys_20k = y_t_20k.squeeze().numpy() * ystd + y0

# Single step at 1kHz
x_ic_1k = torch.zeros(1, nxd, dtype=torch.float32)
x_ic_1k[0, :NX_PHYS] = torch.tensor(x_norm_0, dtype=torch.float32)
u_norm_1k_ic = (u_1k[:1] - u_mean.flatten()) / std_u.flatten()
with torch.no_grad():
    y_t_1k, x_ic_1k = ic_1k.forward(
        x_ic_1k, torch.tensor(u_norm_1k_ic, dtype=torch.float32))
y_phys_1k = y_t_1k.squeeze().numpy() * ystd + y0

err_2 = np.abs(y_phys_1k - y_phys_20k)
print(f'  max|y_1kHz - y_20kHz| per channel:')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    print(f'    {lbl}: {err_2[ch]:.3e}')


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Normalization stats comparison
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'Part 3: Normalization stats — 20kHz vs 1kHz')
print(f'{"="*70}')

for key in ['x_mean', 'std_x', 'std_u', 'u_mean', 'ystd']:
    v20 = stats_20k[key]
    v1  = stats_1k[key]
    diff = np.abs(v20 - v1)
    rel  = diff / (np.abs(v20) + 1e-12)
    print(f'  {key}:')
    print(f'    20kHz: {np.array2string(v20, precision=4, suppress_small=True)}')
    print(f'     1kHz: {np.array2string(v1,  precision=4, suppress_small=True)}')
    print(f'    |diff|: {np.array2string(diff, precision=3, suppress_small=True)}')
    print(f'    rel %:  {np.array2string(rel*100, precision=2, suppress_small=True)}')


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Speed benchmark — CPU vs GPU
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'Part 4: Speed benchmark — CPU vs GPU')
print(f'{"="*70}')

if not HAS_CUDA:
    print(f'  CUDA not available — skipping GPU benchmark')
    print(f'  torch.cuda.is_available() = False')
else:
    print(f'  GPU: {torch.cuda.get_device_name(0)}')

# --- 4a: CPU forward step timing ---
def bench_forward_steps_cpu(ic, n_steps, warmup):
    """Time n_steps forward passes on CPU, return ms/step."""
    x = torch.zeros(1, nxd, dtype=torch.float32)
    x[0, :NX_PHYS] = torch.tensor(x_norm_0, dtype=torch.float32)
    u_bench = torch.tensor(
        (u_1k[:n_steps + warmup] - u_mean.flatten()) / std_u.flatten(),
        dtype=torch.float32,
    )

    # Warmup
    with torch.no_grad():
        for t in range(warmup):
            _, x = ic.forward(x, u_bench[t:t+1])

    t0 = time.perf_counter()
    with torch.no_grad():
        for t in range(warmup, warmup + n_steps):
            _, x = ic.forward(x, u_bench[t:t+1])
    elapsed = time.perf_counter() - t0

    return elapsed, elapsed / n_steps * 1000

ic_cpu = build_interconnect(TS_NEW)
elapsed_cpu, ms_cpu = bench_forward_steps_cpu(ic_cpu, BENCH_STEPS, BENCH_WARMUP)
print(f'\n  Forward steps ({BENCH_STEPS} steps, {BENCH_WARMUP} warmup):')
print(f'    CPU: {elapsed_cpu:.3f}s  ({ms_cpu:.2f} ms/step)')

def bench_forward_steps_gpu(ic, n_steps, warmup):
    """Time n_steps forward passes on GPU, return ms/step."""
    dev = torch.device('cuda')
    ic = ic.to(dev)

    x = torch.zeros(1, nxd, dtype=torch.float32, device=dev)
    x[0, :NX_PHYS] = torch.tensor(x_norm_0, dtype=torch.float32, device=dev)
    u_bench = torch.tensor(
        (u_1k[:n_steps + warmup] - u_mean.flatten()) / std_u.flatten(),
        dtype=torch.float32, device=dev,
    )

    # Warmup
    with torch.no_grad():
        for t in range(warmup):
            _, x = ic.forward(x, u_bench[t:t+1])
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for t in range(warmup, warmup + n_steps):
            _, x = ic.forward(x, u_bench[t:t+1])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, elapsed / n_steps * 1000

if HAS_CUDA:
    ic_gpu = build_interconnect(TS_NEW).to('cuda')
    elapsed_gpu, ms_gpu = bench_forward_steps_gpu(ic_gpu, BENCH_STEPS, BENCH_WARMUP)
    print(f'    GPU: {elapsed_gpu:.3f}s  ({ms_gpu:.2f} ms/step)')
    speedup = elapsed_cpu / elapsed_gpu
    winner = 'GPU' if speedup > 1 else 'CPU'
    print(f'    Speedup: {speedup:.2f}x ({winner} is faster)')

# --- 4b: Batch fwd+bwd timing (CPU only, same reason) ---
def bench_batch_fwd_bwd(batch_size, nf, device_name='cpu'):
    """Time one batch forward + backward pass."""
    dev = torch.device(device_name)
    ic = build_interconnect(TS_NEW).to(dev)

    x_batch = torch.randn(batch_size, nxd, device=dev)
    u_batch = torch.randn(nf, batch_size, nu, device=dev)

    # Warmup
    for _ in range(3):
        x_tmp = x_batch.clone()
        loss = torch.tensor(0.0, device=dev)
        for t in range(min(5, nf)):
            y_t, x_tmp = ic.forward(x_tmp, u_batch[t])
            loss = loss + y_t.pow(2).mean()
        loss.backward()
        ic.zero_grad()

    if device_name == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    x_run = x_batch.clone()
    loss = torch.tensor(0.0, device=dev, requires_grad=True)
    for t in range(nf):
        y_t, x_run = ic.forward(x_run, u_batch[t])
        loss = loss + y_t.pow(2).mean()
    loss.backward()
    if device_name == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed

print(f'\n  Batch fwd+bwd (batch_size={BATCH_SIZE}, nf={NF}):')
t_batch_cpu = bench_batch_fwd_bwd(BATCH_SIZE, NF, 'cpu')
print(f'    CPU: {t_batch_cpu:.3f}s')

if HAS_CUDA:
    t_batch_gpu = bench_batch_fwd_bwd(BATCH_SIZE, NF, 'cuda')
    print(f'    GPU: {t_batch_gpu:.3f}s')
    speedup_batch = t_batch_cpu / t_batch_gpu
    winner_batch = 'GPU' if speedup_batch > 1 else 'CPU'
    print(f'    Speedup: {speedup_batch:.2f}x ({winner_batch} is faster)')

est_windows = 16000
est_batches = est_windows // BATCH_SIZE
t_ref = t_batch_gpu if HAS_CUDA else t_batch_cpu
print(f'\n  Estimated per epoch (1kHz, 8 trajs, ~{est_windows} windows):')
print(f'    ~{t_ref * est_batches:.0f}s  ({est_batches} batches × {t_ref:.1f}s)')


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: CPU vs GPU numerical consistency
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'Part 5: CPU vs GPU numerical consistency')
print(f'{"="*70}')

if not HAS_CUDA:
    print(f'  CUDA not available — skipping')
else:
    y_cpu = rollout(build_interconnect(TS_NEW), x0_phys, u_1k, N_STEPS)

    ic_gpu = build_interconnect(TS_NEW).to('cuda')
    x_norm_0_gpu = (x0_phys - x_mean.flatten()) / std_x.flatten()
    u_norm_gpu = (u_1k[:N_STEPS] - u_mean.flatten()) / std_u.flatten()

    x_g = torch.zeros(1, nxd, dtype=torch.float32, device='cuda')
    x_g[0, :NX_PHYS] = torch.tensor(x_norm_0_gpu, dtype=torch.float32, device='cuda')
    u_g = torch.tensor(u_norm_gpu, dtype=torch.float32, device='cuda')

    y_gpu = np.zeros((N_STEPS, ny), dtype=np.float32)
    with torch.no_grad():
        for t in range(N_STEPS):
            y_t, x_g = ic_gpu.forward(x_g, u_g[t:t+1])
            y_gpu[t] = y_t.squeeze().cpu().numpy() * ystd + y0

    err_5 = np.abs(y_cpu - y_gpu)
    print(f'  {N_STEPS}-step rollout max|y_cpu - y_gpu| per channel:')
    for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
        print(f'    {lbl}: {err_5[:, ch].max():.3e}')


# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'Done.')
print(f'{"="*70}')
