"""
verify_data_model_match.py
--------------------------
Verifies that the RK4 physics model (Gantry_State_Block) can recover the
saved output y from the saved input u_total, for both baseline and MSD
multisine data.

Four parallel workers (one per dataset x rate combination):
  1. MSD      @ 20 kHz
  2. MSD      @ FS_NEW (downsampled, same [::D] stride as gantry_interconnect_dynamic.py)
  3. Baseline @ 20 kHz
  4. Baseline @ FS_NEW

This isolates three failure modes:
  - Data saving bug:     model can't match at 20 kHz
  - Downsampling issue:  20 kHz matches but downsampled doesn't
  - Model mismatch:      Gantry_State_Block doesn't match Simulink

Parallelised with multiprocessing.Pool — one process per (dataset, rate) pair,
2 torch threads each.

Outputs (saved to simulations/gantry_subnet/data_model_match/):
  - <traj>_<tag>_<rate>_{rid}.png  : per-trajectory 3-channel overlay
  - summary_{rid}.png              : bar chart across all trajectories
  - results_{rid}.npz              : all numerical results

Usage:
  conda run -n GraduationProject python scripts/gantry/verification/verify_data_model_match.py
"""

import os
import sys
import multiprocessing
import numpy as np
import torch
from scipy.io import loadmat
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model_augmentation.systems.gantry_ss import Cd, Dd, P

## ═══════════════════════════════════════════════════════════════════════════════
## Configuration
## ═══════════════════════════════════════════════════════════════════════════════

# --- Data directories to verify (run both) ---
DATA_DIRS = {
    'msd':      'multisine',
    'baseline': os.path.join('multisine', 'baseline'),
}

# --- Fixed model constants (1:1 from gantry_interconnect_dynamic.py) ---
NX_PHYS = 6   # physical states: q1, q2, q3, dq1, dq2, dq3
nu  = 3
ny  = 3
Y_OP = None   # None = LPV self-scheduled; float = frozen operating point [m]
SEED = 42

# --- Resampling (1:1 from gantry_interconnect_dynamic.py) ---
FS_ORIG = 20000
FS_NEW  = 1000          # must match gantry_interconnect_dynamic.py
D       = FS_ORIG // FS_NEW   # = 20
TS_NEW  = 1.0 / FS_NEW        # = 0.001 s

# --- Parallelisation ---
N_CPUS_PER_WORKER = 2

# --- Dtype (1:1 from gantry_interconnect_dynamic.py) ---
USE_F64  = False
DTYPE_NP = np.float64    if USE_F64 else np.float32
DTYPE_PT = torch.float64 if USE_F64 else torch.float32

run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')

# --- Files (1:1 from gantry_interconnect_dynamic.py) ---
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
VAL_FILE  = 'V1_X_sym_Y_mid_sweep.mat'
TEST_FILE = 'E1_X_sym_anti_Y_low_offset_sweep.mat'

ALL_FILES = TRAIN_FILES + [VAL_FILE, TEST_FILE]

# --- Precomputed constants (picklable for multiprocessing) ---
P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)  # stage -> logical
Cd_np   = Cd.numpy()
Dd_np   = Dd.numpy()


## ═══════════════════════════════════════════════════════════════════════════════
## Data loading (1:1 from gantry_interconnect_dynamic.py)
## ═══════════════════════════════════════════════════════════════════════════════

def _load_u(d):
    """Return plant input: 'u_total' for multisine data, 'u' for trajectory data."""
    if 'u_total' in d:
        return d['u_total']
    return d['u']


## ═══════════════════════════════════════════════════════════════════════════════
## Normalisation (1:1 from gantry_interconnect_dynamic.py, parameterised by rate)
## ═══════════════════════════════════════════════════════════════════════════════

def compute_norm(traj_dir, stride, fs_target):
    """Compute normalisation stats — same pipeline as gantry_interconnect_dynamic.py."""
    train_u, train_y = [], []
    for f in TRAIN_FILES:
        d = loadmat(os.path.join(traj_dir, f), squeeze_me=True)
        train_u.append(_load_u(d)[::stride].astype(DTYPE_NP))
        train_y.append(d['y'][::stride].astype(DTYPE_NP))

    u_all = np.concatenate(train_u)
    y_all = np.concatenate(train_y)

    fs = fs_target
    x_logical_list = []
    for y_tr in train_y:
        pos_logical = (P_inv_T @ y_tr.T).T        # (N, 3) stage -> logical
        vel_logical = np.diff(pos_logical, axis=0) * fs  # (N-1, 3)
        vel_logical = np.vstack([vel_logical[:1], vel_logical])  # (N, 3)
        x_logical_list.append(np.hstack([pos_logical, vel_logical]))  # (N, 6)
    x_all = np.concatenate(x_logical_list)

    x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
    std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
    std_u  = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
    u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
    ystd   = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    y0     = (Cd_np @ x_mean.flatten()).astype(DTYPE_NP)

    # Cd_norm[i,j] = Cd[i,j] * std_x[j] / ystd[i]
    Cd_norm = Cd_np * std_x.flatten()[None, :] / ystd[:, None]

    return dict(x_mean=x_mean, std_x=std_x, std_u=std_u, u_mean=u_mean,
                ystd=ystd, y0=y0, Cd_norm=Cd_norm)


## ═══════════════════════════════════════════════════════════════════════════════
## Physics model (same structure as build_physics_only in diagnose_encoder.py,
## same Gantry_State_Block args as gantry_interconnect_dynamic.py)
## ═══════════════════════════════════════════════════════════════════════════════

def build_physics_ic(norm, Ts):
    """Build a physics-only interconnect (nxd=6, no ANN)."""
    from model_augmentation.utils.utils import selection_matrix, expansion_matrix
    from model_augmentation.fit_systems.interconnect import Interconnect
    from model_augmentation.fit_systems.blocks import (
        Gantry_State_Block, Linear_Output_Block,
    )

    ic = Interconnect(NX_PHYS, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=norm['std_x'], std_u=norm['std_u'],
        x_mean=norm['x_mean'], u_mean=norm['u_mean'], Ts=Ts,
    ).to(DTYPE_PT)
    out_block = Linear_Output_Block(C=norm['Cd_norm'], D=Dd_np)

    ic.add_block(phy_block)
    ic.add_block(out_block)
    ic.connect_signals("x", phy_block)
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp")
    ic.connect_signals("x", out_block)
    ic.connect_block_signals(out_block, ["u"], ["y"])
    return ic


## ═══════════════════════════════════════════════════════════════════════════════
## Rollout
## ═══════════════════════════════════════════════════════════════════════════════

def analytical_x0(y_data, dt):
    """Derive physical x0 from first two output samples."""
    pos = P_inv_T @ y_data[0]
    vel = P_inv_T @ ((y_data[1] - y_data[0]) / dt)
    return np.concatenate([pos, vel])


def rollout(ic, u_raw, y_raw, norm, dt):
    """Full rollout of physics-only model. Returns y_hat (N, 3) in physical units."""
    # Normalise input (same as gantry_interconnect_dynamic.py pipeline)
    u_norm = (u_raw - norm['u_mean'].flatten()) / norm['std_u'].flatten()

    # Analytical x0
    x0_phys = analytical_x0(y_raw, dt)
    x0_norm = ((x0_phys.reshape(NX_PHYS, 1) - norm['x_mean']) / norm['std_x']).flatten()

    x = torch.tensor(x0_norm.reshape(1, -1), dtype=DTYPE_PT)
    u_t = torch.tensor(u_norm.astype(DTYPE_NP), dtype=DTYPE_PT)

    y_list = []
    with torch.no_grad():
        for t in range(len(u_t)):
            y_t, x = ic(x, u_t[t:t + 1])
            y_list.append(y_t.squeeze(0).numpy())

    y_hat_norm = np.array(y_list)
    return y_hat_norm * norm['ystd'] + norm['y0']


## ═══════════════════════════════════════════════════════════════════════════════
## Worker function (runs in spawned process)
## ═══════════════════════════════════════════════════════════════════════════════

def process_worker(args):
    """Worker: process all trajectories for one (dataset, rate) combination."""
    import time as _time
    tag, subdir, rate_tag, stride, fs_target, Ts, traj_dir, save_dir, rid, cpus = args
    torch.set_num_threads(cpus)
    t0 = _time.time()

    worker_id = f"{tag}_{rate_tag}"

    # Compute normalisation for this rate
    norm = compute_norm(traj_dir, stride=stride, fs_target=fs_target)

    # Build physics model for this rate
    ic = build_physics_ic(norm, Ts=Ts)

    results = []
    ch_labels = ['X1', 'X2', 'Y']

    for fname in ALL_FILES:
        fpath = os.path.join(traj_dir, fname)
        if not os.path.isfile(fpath):
            continue

        d = loadmat(fpath, squeeze_me=True)
        u_raw = _load_u(d)[::stride].astype(DTYPE_NP)
        y_raw = d['y'][::stride].astype(DTYPE_NP)
        label = os.path.splitext(fname)[0]

        y_hat = rollout(ic, u_raw, y_raw, norm, Ts)
        N = min(len(y_hat), len(y_raw))
        err = y_hat[:N] - y_raw[:N]
        rmse = np.sqrt(np.mean(err ** 2, axis=0))
        nrms = rmse / norm['ystd']

        results.append({
            'label': label,
            'rmse': rmse.copy(),
            'nrms': nrms.copy(),
            'y_hat': y_hat[:N].copy(),
            'y_ref': y_raw[:N].copy(),
        })

        # Per-trajectory plot: 3-channel overlay
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        t_plot = np.arange(N) * Ts
        for ch in range(3):
            ax = axes[ch]
            ax.plot(t_plot, y_raw[:N, ch], 'k', lw=0.5, label='Saved data')
            ax.plot(t_plot, y_hat[:N, ch], 'C0', lw=0.5, alpha=0.8,
                    label=f'RK4 (NRMS={nrms[ch]:.4f})')
            ax.set_ylabel(f'{ch_labels[ch]} [m]', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time [s]')
        fig.suptitle(f'{label} [{worker_id}]', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{label}_{worker_id}_{rid}.png'), dpi=150)
        plt.close(fig)

    elapsed = _time.time() - t0
    return {
        'worker_id': worker_id,
        'tag': tag,
        'rate_tag': rate_tag,
        'fs': fs_target,
        'stride': stride,
        'results': results,
        'elapsed': elapsed,
    }


## ═══════════════════════════════════════════════════════════════════════════════
## Main
## ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    base_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                 'data', 'gantry', 'matlab')
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                            'simulations', 'gantry_subnet', 'data_model_match')
    os.makedirs(save_dir, exist_ok=True)

    print(f"Verification: physics model vs saved data")
    print(f"FS_ORIG={FS_ORIG}  FS_NEW={FS_NEW}  D={D}")
    print(f"CPUs per worker: {N_CPUS_PER_WORKER}")
    print(f"Run ID: {run_id}\n")

    # Build worker args: one per (dataset, rate) combination
    worker_args = []
    for tag, subdir in DATA_DIRS.items():
        traj_dir = os.path.join(base_data_dir, subdir)
        if not os.path.isdir(traj_dir):
            print(f"[{tag}] Directory not found: {traj_dir} — skipping.")
            continue

        # Full rate (20 kHz)
        worker_args.append((
            tag, subdir, f'{FS_ORIG}Hz', 1, FS_ORIG, 1.0/FS_ORIG,
            traj_dir, save_dir, run_id, N_CPUS_PER_WORKER,
        ))
        # Downsampled
        worker_args.append((
            tag, subdir, f'{FS_NEW}Hz', D, FS_NEW, TS_NEW,
            traj_dir, save_dir, run_id, N_CPUS_PER_WORKER,
        ))

    n_workers = len(worker_args)
    print(f"Launching {n_workers} workers in parallel...\n")
    for wa in worker_args:
        print(f"  {wa[0]}_{wa[2]}")

    with multiprocessing.Pool(n_workers) as pool:
        all_worker_results = pool.map(process_worker, worker_args)

    # ── Print results ─────────────────────────────────────────────────────
    ch_labels = ['X1', 'X2', 'Y']

    for wr in all_worker_results:
        m, s = int(wr['elapsed'] // 60), int(wr['elapsed'] % 60)
        print(f"\n{'='*70}")
        print(f"  {wr['worker_id']}  ({m}m{s:02d}s)")
        print(f"{'='*70}")
        print(f"  {'File':<42s}  "
              f"{'X1 RMSE':>10s}  {'X2 RMSE':>10s}  {'Y RMSE':>10s}  "
              f"{'X1 NRMS':>8s}  {'X2 NRMS':>8s}  {'Y NRMS':>8s}")
        for r in wr['results']:
            print(f"  {r['label']:<42s}  "
                  f"{r['rmse'][0]:10.2e}  {r['rmse'][1]:10.2e}  {r['rmse'][2]:10.2e}  "
                  f"{r['nrms'][0]:8.4f}  {r['nrms'][1]:8.4f}  {r['nrms'][2]:8.4f}")

    # ── Summary bar chart: all 4 workers side by side ─────────────────────
    # Group by dataset tag
    tags_seen = list(dict.fromkeys(wr['tag'] for wr in all_worker_results))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    n_workers_total = len(all_worker_results)
    bar_w = 0.8 / n_workers_total

    # Use first worker's labels as reference (all should have same trajectories)
    ref_labels = [r['label'] for r in all_worker_results[0]['results']]
    x_pos = np.arange(len(ref_labels))

    colors = {'msd': 'C0', 'baseline': 'C1'}
    alphas = {}
    for wr in all_worker_results:
        alphas[wr['worker_id']] = 1.0 if str(FS_ORIG) in wr['rate_tag'] else 0.5

    for ch in range(3):
        ax = axes[ch]
        for bi, wr in enumerate(all_worker_results):
            vals = [r['rmse'][ch] for r in wr['results']]
            color = colors.get(wr['tag'], f'C{bi}')
            alpha = alphas.get(wr['worker_id'], 1.0)
            ax.bar(x_pos + bi * bar_w, vals[:len(x_pos)], bar_w,
                   label=wr['worker_id'], color=color, alpha=alpha)
        ax.set_xticks(x_pos + bar_w * (n_workers_total - 1) / 2)
        ax.set_xticklabels(ref_labels, rotation=45, ha='right', fontsize=5)
        ax.set_ylabel('RMSE [m]')
        ax.set_title(ch_labels[ch])
        ax.grid(True, alpha=0.3, axis='y')
        if ch == 0:
            ax.legend(fontsize=6)

    fig.suptitle(f'Physics model vs saved data — RMSE comparison', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'summary_{run_id}.png'), dpi=150)
    plt.close(fig)
    print(f"\nSaved: summary_{run_id}.png")

    print("\nDone.")
