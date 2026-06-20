"""Memory diagnostic for gantry_interconnect_dynamic.py (OOM on 64 GB server).

Self-contained: builds its own models and data, runs in minutes, needs no prior
training artifacts. Follows Jan's diagnostic tips:
  Stage 1: data-segment accounting (size of make_training_data output vs analytic formula)
  Stage 2: training-step memory scaling with batch_size / nf / up_sample (BPTT graph)
  Stage 3: leak check (RSS + wall time per epoch, tiny config)
  Stage 4: vanilla deepSI SS_encoder baseline on the same data (copy-vs-view windowing)
  Stage 5: data scaling with number of trajectories

Usage:
  conda run -n GraduationProject python scripts/gantry/verification/_memory_diagnostic.py --stage all
  (or --stage 1 ... --stage 5)

Background (code references):
  - model_augmentation/utils/deepSI_corrections.py:8-26  fixed_System_data.to_hist_future_data
    materializes every overlapping window as a real copy (vanilla deepSI stride=1 uses
    zero-copy sliding_window_view, deepSI/system_data/system_data.py:305-315).
  - model_augmentation/fit_systems/interconnect.py:432-438  SSE_Interconnect.loss rolls the
    interconnect nf steps with the full autograd graph alive: memory ~ batch_size * nf.
"""

import os
import sys
import gc
import time
import argparse
import numpy as np
import torch
import deepSI
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model_augmentation.utils.utils import *
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.systems.gantry_ss import Cd, Dd, P

## ═══════════════════════════════════════════════════════════════════════════════
## RSS measurement (no psutil in env: /proc on Linux, psapi on Windows)
## ═══════════════════════════════════════════════════════════════════════════════

def rss_mb():
    """Resident set size of this process in MB."""
    if sys.platform.startswith('linux'):
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0  # kB -> MB
        return float('nan')
    elif sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD),
                        ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t),
                        ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t),
                        ('PeakPagefileUsage', ctypes.c_size_t)]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        fn = ctypes.windll.kernel32.K32GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        ok = fn(ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize / 2**20 if ok else float('nan')
    return float('nan')


def gc_baseline():
    gc.collect()
    return rss_mb()


## ═══════════════════════════════════════════════════════════════════════════════
## Configuration (mirrors the failing server run: trajectories, default encoder)
## ═══════════════════════════════════════════════════════════════════════════════

NX_PHYS = 6
nu = 3
ny = 3
SEED = 42
FS_ORIG = 20000
FS_NEW = 4000
D = FS_ORIG // FS_NEW
TS_NEW = 1.0 / FS_NEW
DTYPE_NP = np.float32
DTYPE_PT = torch.float32

# Failing server config (for extrapolation reference)
SERVER_HP = dict(nf=1200, na_nb=400, batch_size=4000, n_nodes_per_layer=64,
                 n_hidden_layers=2, NX_ANN=2, up_sample=2)

TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'data', 'gantry', 'matlab', 'trajectories')
TRAIN_FILES = [
    'T1_Y_sweep_conservative.mat', 'T2_X_sym_Y030.mat', 'T3_X_sym_Y000.mat',
    'T4_X_antisym_Y020.mat', 'T5_X_sym_Y_sweep.mat', 'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat', 'T8_X_sym_anti_Y_sweep.mat',
]
VAL_FILE = 'V1_X_sym_Y_mid_sweep.mat'


def load_traj(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = d['u_total'] if 'u_total' in d else d['u']
    return deepSI.System_data(u=u[::D].astype(DTYPE_NP),
                              y=d['y'][::D].astype(DTYPE_NP), dt=TS_NEW)


print('Loading trajectories...')
train_list = [load_traj(f) for f in TRAIN_FILES]
train_data = deepSI.System_data_list(train_list)
val_data = load_traj(VAL_FILE)
N_per_traj = [len(t.u) for t in train_list]
print(f'  {len(train_list)} trajectories, samples each: {N_per_traj}')
print(f'  RSS after data load: {rss_mb():.0f} MB')

# Normalisation constants (same construction as gantry_interconnect_dynamic.py)
u_all = np.concatenate([t.u for t in train_list])
y_all = np.concatenate([t.y for t in train_list])
fs = 1.0 / train_list[0].dt
P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)
x_logical_list = []
for t in train_list:
    pos_logical = (P_inv_T @ t.y.T).T
    vel_logical = np.diff(pos_logical, axis=0) * fs
    vel_logical = np.vstack([vel_logical[:1], vel_logical])
    x_logical_list.append(np.hstack([pos_logical, vel_logical]))
x_all = np.concatenate(x_logical_list)
x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
std_x = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
std_u = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
ystd = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
y0 = (Cd.numpy() @ x_mean.flatten()).astype(DTYPE_NP)
Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
Dd_np = Dd.numpy()
PHY_IX = np.arange(NX_PHYS)


def build_model(hp):
    """Minimal copy of gantry_interconnect_dynamic.build_model (default encoder)."""
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    na = nb = hp['na_nb']

    ic = Interconnect(nxd, nu, ny, debugging=False)
    phy_block = Gantry_State_Block(
        Y_op=None, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
        up_sample=hp['up_sample'],
    ).to(DTYPE_PT)
    out_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    ic.add_block(phy_block)
    ic.add_block(out_block)
    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=nxd,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=zero_init_feed_forward_nn, activation=torch.nn.Tanh,
    )
    ic.add_block(ann_block)
    ic.connect_block_signals(ann_block, ["x", "u"], ["xp"])
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", out_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_block, ["u"], ["y"])

    fit_sys = SSE_Interconnect(
        interconnect=ic, na=na, nb=nb,
        e_net_kwargs={"n_nodes_per_layer": hp['n_nodes_per_layer'],
                      "n_hidden_layers": hp['n_hidden_layers']},
    )
    fit_sys.norm.u0 = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0 = y0
    fit_sys.norm.ystd = ystd
    fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
    for net in (fit_sys.encoder, fit_sys.hfn):
        net.to(DTYPE_PT)
    return fit_sys


def predicted_window_bytes(na, nb, nf, stride, n_samples_list):
    """Analytic size of the materialized training arrays.

    Mirrors fixed_System_data.to_hist_future_data (deepSI_corrections.py:16-24):
    windows per trajectory = len(range(k0+k0_right, N+1, stride)) with k0=max(na,nb),
    k0_right=nf; bytes per window = (nb*nu + na*ny + nf*(nu+ny)) * 4 (float32).
    """
    k0 = max(na, nb)
    n_windows = sum(len(range(k0 + nf, N + 1, stride)) for N in n_samples_list)
    bytes_per_window = (nb * nu + na * ny + nf * (nu + ny)) * 4
    return n_windows, n_windows * bytes_per_window


## ═══════════════════════════════════════════════════════════════════════════════
## Stage 1: data-segment accounting
## ═══════════════════════════════════════════════════════════════════════════════

def stage1():
    print('\n' + '=' * 78)
    print('STAGE 1: data-segment accounting (make_training_data size)')
    print('=' * 78)
    hp = dict(SERVER_HP)
    raw_mb = sum(t.u.nbytes + t.y.nbytes for t in train_list) / 2**20
    print(f'Raw training data (u+y, float32): {raw_mb:.1f} MB')
    print(f'{"na=nb":>6} {"nf":>6} {"stride":>7} {"windows":>9} '
          f'{"predicted":>10} {"nbytes":>10} {"RSS delta":>10}')

    fit_sys = build_model({**hp, 'na_nb': hp['na_nb']})
    for na_nb, nf in [(400, 25), (400, 100), (400, 400), (400, 1200), (100, 400)]:
        for stride in [1, max(1, nf // 2)]:
            if stride != 1 and nf == 25:
                continue
            fit_sys.na = fit_sys.nb = na_nb
            base = gc_baseline()
            data = fit_sys.make_training_data(
                fit_sys.norm.transform(train_data), nf=nf, stride=stride)
            actual = sum(d.nbytes for d in data) / 2**20
            after = rss_mb()
            n_win, pred = predicted_window_bytes(na_nb, na_nb, nf, stride, N_per_traj)
            print(f'{na_nb:>6} {nf:>6} {stride:>7} {n_win:>9} '
                  f'{pred/2**20:>9.0f}M {actual:>9.0f}M {after-base:>9.0f}M')
            del data
            gc.collect()
    del fit_sys
    gc.collect()
    print('\nInterpretation: nbytes ~ predicted -> the 1.6 GB print is fully explained')
    print('by stride-1 overlapping windows; expansion factor vs raw data is shown above.')


## ═══════════════════════════════════════════════════════════════════════════════
## Stage 2: training-step memory scaling (BPTT graph)
## ═══════════════════════════════════════════════════════════════════════════════

def one_training_step(fit_sys, batch, nf):
    """Forward + backward on one batch; return (RSS after forward, RSS after backward)."""
    fit_sys.train()
    fit_sys.optimizer.zero_grad()
    loss = fit_sys.loss(*batch, nf=nf)
    rss_fwd = rss_mb()           # graph fully alive here
    loss.backward()
    fit_sys.optimizer.step()
    rss_bwd = rss_mb()
    return rss_fwd, rss_bwd


def measure_step_subprocess(batch_size, nf, up_sample, na_nb=SERVER_HP['na_nb'],
                            hold_full_data=0):
    """Run a single config in a fresh process and report true peak RSS.

    A fresh process is required: within one process the allocator reuses memory
    freed after a previous step without returning it to the OS, so RSS deltas of
    later steps read ~0 (observed on Linux). A sampling thread captures the peak
    (forward graph + backward overhead) during the step.
    """
    import threading

    hp = dict(SERVER_HP, na_nb=na_nb, batch_size=batch_size, nf=nf, up_sample=up_sample)
    torch.manual_seed(SEED)
    fit_sys = build_model(hp)

    # Tiny warm-up triggers lazy inits (connection matrices, Adam state) cheaply
    warm = make_batch(fit_sys, 5, 2)
    one_training_step(fit_sys, warm, 5)
    del warm

    if hold_full_data:
        # Replicate the real run: full stride-1 training array + dataloader tensors
        # held in memory while the step executes (as in fit(), interconnect.py:566-589)
        data_full = fit_sys.make_training_data(
            fit_sys.norm.transform(train_data), nf=nf, stride=1)
        held = [torch.as_tensor(d, dtype=DTYPE_PT) for d in data_full]
        ids = np.random.permutation(len(data_full[0]))[:batch_size]
        batch = [d[ids] for d in held]
        print(f'holding full data: {sum(d.nbytes for d in data_full)/2**20:.0f} MB')
    else:
        batch = make_batch(fit_sys, nf, batch_size)
    base = gc_baseline()

    peak = [base]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], rss_mb())
            time.sleep(0.005)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.time()
    one_training_step(fit_sys, batch, nf)
    dt_step = time.time() - t0
    stop.set()
    th.join()
    peak[0] = max(peak[0], rss_mb())

    print(f'MEASURE_RESULT base={base:.1f} peak={peak[0]:.1f} t={dt_step:.2f}')


def make_batch(fit_sys, nf, batch_size):
    """Build exactly ~batch_size windows (large stride keeps the data array small)."""
    na = fit_sys.na
    N_total = sum(max(0, N - max(na, fit_sys.nb) - nf + 1) for N in N_per_traj)
    stride = max(1, N_total // batch_size)
    data = fit_sys.make_training_data(
        fit_sys.norm.transform(train_data), nf=nf, stride=stride)
    n = min(batch_size, len(data[0]))
    return [torch.as_tensor(d[:n], dtype=DTYPE_PT) for d in data]


def stage2():
    print('\n' + '=' * 78)
    print('STAGE 2: training-step memory scaling (batch_size x nf x up_sample)')
    print('  each config runs in a fresh subprocess; peak RSS sampled during the step')
    print('=' * 78)
    import subprocess
    import re

    configs = (
        [(b, 100, 2) for b in (64, 256, 1000)] +
        [(256, f, 2) for f in (25, 400)] +
        [(256, 100, 1)]
    )

    print(f'{"batch":>6} {"nf":>5} {"up_s":>5} {"peak-base MB":>13} '
          f'{"kB/(batch*nf)":>14} {"t/step [s]":>11}')
    results = []
    for (b, f, up) in configs:
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             '--measure-step', str(b), str(f), str(up), '100', '0'],
            capture_output=True, text=True)
        m = re.search(r'MEASURE_RESULT base=([\d.]+) peak=([\d.]+) t=([\d.]+)',
                      out.stdout)
        if m is None:
            print(f'{b:>6} {f:>5} {up:>5}  FAILED (see stderr below)')
            print(out.stdout[-2000:])
            print(out.stderr[-2000:])
            continue
        base, peak, dt_step = map(float, m.groups())
        delta = peak - base
        per_unit = delta / (b * f) * 1000  # kB per window-step
        results.append(((b, f, up), delta, per_unit))
        print(f'{b:>6} {f:>5} {up:>5} {delta:>13.0f} {per_unit:>13.3f} {dt_step:>11.1f}')

    if results:
        # Extrapolate to the failing server config from the largest measured config
        (b_ref, f_ref, _), delta_ref, _ = max(results, key=lambda r: r[0][0] * r[0][1])
        scale = (SERVER_HP['batch_size'] * SERVER_HP['nf']) / (b_ref * f_ref)
        print(f'\nExtrapolation (linear in batch*nf) to server config '
              f'batch={SERVER_HP["batch_size"]}, nf={SERVER_HP["nf"]}:')
        print(f'  estimated step peak (graph + backward) ~ {delta_ref * scale / 1024:.1f} GB '
              f'(+ {predicted_window_bytes(SERVER_HP["na_nb"], SERVER_HP["na_nb"], SERVER_HP["nf"], 1, N_per_traj)[1] / 2**30:.1f} GB data array + model/optimizer)')


## ═══════════════════════════════════════════════════════════════════════════════
## Stage 3: leak check (RSS and wall time per epoch)
## ═══════════════════════════════════════════════════════════════════════════════

def stage3(n_epochs=15):
    print('\n' + '=' * 78)
    print(f'STAGE 3: leak check ({n_epochs} epochs, tiny config)')
    print('=' * 78)
    hp = dict(SERVER_HP, nf=50, na_nb=50, batch_size=64, n_nodes_per_layer=16)
    torch.manual_seed(SEED)
    fit_sys = build_model(hp)
    print(f'{"epoch":>6} {"RSS MB":>8} {"t [s]":>7}')
    for ep in range(n_epochs):
        t0 = time.time()
        fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data,
                    batch_size=hp['batch_size'], epochs=1, auto_fit_norm=False,
                    loss_kwargs={'nf': hp['nf'], 'stride': 20},
                    optimizer_kwargs={'lr': 5e-4},
                    validation_measure='sim-RMS', verbose=0)
        gc.collect()
        print(f'{ep:>6} {rss_mb():>8.0f} {time.time()-t0:>7.1f}')
    del fit_sys
    gc.collect()
    print('\nInterpretation: flat RSS and flat epoch time -> no leak;')
    print('monotone growth -> leak (then bisect training loop vs validation).')


## ═══════════════════════════════════════════════════════════════════════════════
## Stage 4: vanilla deepSI black-box baseline (Jan's tip)
## ═══════════════════════════════════════════════════════════════════════════════

def stage4():
    print('\n' + '=' * 78)
    print('STAGE 4: vanilla deepSI SS_encoder on the same data')
    print('=' * 78)
    from deepSI.fit_systems.encoders import SS_encoder
    na = nb = SERVER_HP['na_nb']
    nf = SERVER_HP['nf']

    sys_bb = SS_encoder(nx=NX_PHYS + SERVER_HP['NX_ANN'], na=na, nb=nb)
    sys_bb.norm.fit(train_data)

    base = gc_baseline()
    data_bb = sys_bb.make_training_data(sys_bb.norm.transform(train_data), nf=nf)
    nbytes_bb = sum(d.nbytes for d in data_bb) / 2**20
    rss_bb = rss_mb() - base
    print(f'vanilla deepSI:    nbytes print = {nbytes_bb:.0f} MB, actual RSS delta = {rss_bb:.0f} MB')
    print('  (single System_data windows would be sliding_window_view = views, but')
    print('   System_data_list.to_hist_future_data np.concatenates the trajectories')
    print('   -> materialized copies in vanilla deepSI too; deepSI system_data.py:682)')
    del data_bb
    gc.collect()

    fit_sys = build_model(dict(SERVER_HP))
    base = gc_baseline()
    data_ma = fit_sys.make_training_data(fit_sys.norm.transform(train_data), nf=nf)
    nbytes_ma = sum(d.nbytes for d in data_ma) / 2**20
    rss_ma = rss_mb() - base
    print(f'model_augmentation: nbytes print = {nbytes_ma:.0f} MB, actual RSS delta = {rss_ma:.0f} MB')
    print('  (fixed_System_data.to_hist_future_data materializes real copies)')
    del data_ma, fit_sys
    gc.collect()

    print('\nInterpretation (measured): both pipelines materialize ~ the same RSS for')
    print('multi-trajectory data, so the deepSI_corrections.py override is NOT the')
    print('differentiator. The 1.6+ GB is inherent to stride-1 windowing with large')
    print('na/nb/nf. Remedies: stride>1, smaller nf, or online_construct=True (the')
    print('latter requires adding hist_future_dataset support to the override,')
    print('mirroring deepSI system_data.py:302-303).')


## ═══════════════════════════════════════════════════════════════════════════════
## Stage 5: scaling with number of trajectories
## ═══════════════════════════════════════════════════════════════════════════════

def stage5():
    print('\n' + '=' * 78)
    print('STAGE 5: data array size vs number of trajectories (nf=400, na=nb=100)')
    print('=' * 78)
    hp = dict(SERVER_HP, na_nb=100)
    fit_sys = build_model(hp)
    print(f'{"n_traj":>7} {"windows":>9} {"nbytes MB":>10} {"RSS delta":>10}')
    for n_traj in (1, 2, 4, 8):
        subset = deepSI.System_data_list(train_list[:n_traj])
        base = gc_baseline()
        data = fit_sys.make_training_data(fit_sys.norm.transform(subset), nf=400)
        actual = sum(d.nbytes for d in data) / 2**20
        after = rss_mb()
        print(f'{n_traj:>7} {len(data[0]):>9} {actual:>10.0f} {after-base:>10.0f}')
        del data
        gc.collect()
    del fit_sys
    gc.collect()
    print('\nInterpretation: linear in n_traj confirms no superlinear data effect.')


## ═══════════════════════════════════════════════════════════════════════════════
## Main
## ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', default='all', choices=['1', '2', '3', '4', '5', 'all'])
    parser.add_argument('--measure-step', nargs='+', type=int, default=None,
                        metavar='BATCH NF UP_SAMPLE [NA_NB] [HOLD_FULL_DATA]',
                        help='single-step peak measurement in a fresh process; '
                             'NA_NB defaults to 400, HOLD_FULL_DATA=1 also keeps the '
                             'full stride-1 training array in memory like the real run')
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if args.measure_step is not None:
        measure_step_subprocess(*args.measure_step)
        sys.exit(0)

    stages = {'1': stage1, '2': stage2, '3': stage3, '4': stage4, '5': stage5}
    to_run = list(stages.values()) if args.stage == 'all' else [stages[args.stage]]
    for s in to_run:
        s()

    print('\nDone.')
