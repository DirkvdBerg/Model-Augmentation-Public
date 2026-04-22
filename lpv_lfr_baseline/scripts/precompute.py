"""
precompute.py
-------------
One-time setup: computes and caches all data that does not depend on trainable
parameters. Writes a single .pt cache file. Subsequent calls return the cached
result immediately.

Computes
--------
trajs                    : list of dicts — per-trajectory u (1,T,3), q1 (T,3),
                           state_traj (T,6), and metadata (id, group, file, N, fs)
sigma                    : dict traj_id -> (3,) — per-trajectory per-channel
                           output std for loss normalization, clamped to min 1e-4.
                           All three channels contribute (no masks).
rmse_entries             : list of dicts — per-trajectory detuned-model RMSE
rmse_baseline_normalized : float — group-balanced RMSE baseline in sigma units
segment_len              : int — BPTT window length chosen by segment diagnostic

All tensors are on CPU in the requested dtype. The training loop moves them to
the training device via .to(device).

What is NOT computed here
-------------------------
G, alpha, beta, gamma, N0, N1, N2 — all depend on trainable parameters and must
be rebuilt each forward pass from current log_params.
"""

import os
from pathlib import Path

import torch
from scipy.io import loadmat

from lpv_lfr_baseline.core.physics import P as _P, ts as _ts
from lpv_lfr_baseline.scripts.data_utils import compute_rmse_baseline_metrics

CACHE_VERSION = 1


# ----------------------------------------------------------------------
# Cache helpers
# ----------------------------------------------------------------------

def _fingerprint(traj_specs, dtype):
    """Stable cache key — invalidated by any change to traj set or dtype."""
    return (
        CACHE_VERSION,
        tuple((s['id'], s['file'], s['group']) for s in traj_specs),
        str(dtype),
    )


# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------

def _load_trajectory(mat_path, dtype):
    """Load one .mat trajectory. Returns (u, q1, fs)."""
    mat = loadmat(mat_path)
    u   = torch.tensor(mat['u_q1'], dtype=dtype).unsqueeze(0)   # (1, T, 3)
    q1  = torch.tensor(mat['q1'],   dtype=dtype)                # (T, 3)
    fs  = float(mat['fs'].squeeze()) if 'fs' in mat else None
    return u, q1, fs


def _build_state_traj_logical(q1_stage, P, ts_val, dtype):
    """
    Build [q; qdot] in logical coordinates from stage-position data.

    Parameters
    ----------
    q1_stage : (T, 3) stage positions
    P        : (3, 3) stage <-> logical transform
    ts_val   : float  sample period [s]
    dtype    : torch dtype

    Returns
    -------
    (T, 6) tensor  [q_logical; qdot_logical]
    """
    q_logical = torch.linalg.solve(P.to(dtype).T, q1_stage.to(dtype).T).T   # (T, 3)

    qdot = torch.empty_like(q_logical)
    if q_logical.shape[0] == 1:
        qdot.zero_()
    else:
        qdot[0]    = (q_logical[1]  - q_logical[0])  / ts_val
        qdot[1:-1] = (q_logical[2:] - q_logical[:-2]) / (2 * ts_val)
        qdot[-1]   = (q_logical[-1] - q_logical[-2])  / ts_val

    return torch.cat([q_logical, qdot], dim=-1)   # (T, 6)


def _compute_sigma(trajs, dtype):
    """
    Per-trajectory per-channel output std, clamped to min 1e-4.

    All three channels contribute — no channel masks. Channels with
    near-zero variance (controller-suppressed) are clamped so they
    produce finite but small loss signal rather than division by zero.
    """
    sigma = {}
    for traj in trajs:
        ch = torch.empty(3, dtype=dtype)
        for c in range(3):
            ch[c] = traj['q1'][:, c].std().clamp(min=1e-4)
        sigma[traj['id']] = ch
    return sigma


# ----------------------------------------------------------------------
# RMSE aggregation helpers
# ----------------------------------------------------------------------

def _aggregate_grouped_rmse(entries):
    """Group-balanced RMSE scalar from a list of per-trajectory MSE entries."""
    if len(entries) == 1:
        return float(entries[0]['rmse_total'])
    group_mse = {}
    for e in entries:
        group_mse.setdefault(e['group'], []).append(e['mse_total'])
    overall_mse = sum(sum(v) / len(v) for v in group_mse.values()) / len(group_mse)
    return overall_mse ** 0.5


def _aggregate_normalized_rmse_baseline(rmse_entries, sigma):
    """
    Group-balanced RMSE baseline in sigma-normalized (dimensionless) units.

    Uses float64 throughout for the scalar computation regardless of training
    dtype — normalization constants should be computed at full precision.
    """
    def _normalized_mse(entry):
        rmse_ch = torch.tensor(entry['rmse_ch'], dtype=torch.float64)
        s = sigma[entry['id']].to(torch.float64)
        return (rmse_ch / s).pow(2).mean().item()

    if len(rmse_entries) == 1:
        return _normalized_mse(rmse_entries[0]) ** 0.5

    group_mse = {}
    for e in rmse_entries:
        group_mse.setdefault(e['group'], []).append(_normalized_mse(e))
    overall_mse = sum(sum(v) / len(v) for v in group_mse.values()) / len(group_mse)
    return overall_mse ** 0.5


# ----------------------------------------------------------------------
# Core computation
# ----------------------------------------------------------------------

def _compute(traj_specs, traj_dir, save_dir, dtype):
    """Run all precomputation. Only called when cache is absent or stale."""
    P      = _P.to(dtype)
    ts_val = float(_ts)

    # --- trajectories and state trajectories ---
    print('  precompute: loading trajectories')
    trajs = []
    for spec in traj_specs:
        mat_path = os.path.join(traj_dir, spec['file'])
        u, q1, fs = _load_trajectory(mat_path, dtype)
        state_traj = _build_state_traj_logical(q1, P, ts_val, dtype)
        trajs.append({
            'id':         spec['id'],
            'group':      spec['group'],
            'file':       spec['file'],
            'N':          int(q1.shape[0]),
            'fs':         fs,
            'u':          u,            # (1, T, 3) CPU
            'q1':         q1,           # (T, 3)    CPU
            'state_traj': state_traj,   # (T, 6)    CPU
        })
        print(f'    {spec["id"]}: T={q1.shape[0]}, fs={fs} Hz')

    # --- sigma ---
    print('  precompute: computing sigma')
    sigma = _compute_sigma(trajs, dtype)
    for traj in trajs:
        s = sigma[traj['id']]
        print(f'    {traj["id"]}: sigma = [{s[0]:.3e}, {s[1]:.3e}, {s[2]:.3e}]')

    # --- RMSE baseline (detuned initial model) ---
    print('  precompute: computing RMSE baseline')
    rmse_entries = []
    for traj in trajs:
        mat_path = os.path.join(traj_dir, traj['file'])
        metrics = compute_rmse_baseline_metrics(
            mat_path=mat_path,
            x0_logical=traj['state_traj'][:1].cpu(),
            verbose=False,
        )
        rmse_entries.append({
            'id':         traj['id'],
            'group':      traj['group'],
            'file':       traj['file'],
            'mse_total':  metrics['mse_total'],
            'rmse_total': metrics['rmse_total'],
            'rmse_ch':    metrics['rmse_ch'],
        })
        print(f'    {traj["id"]}: RMSE = {metrics["rmse_total"]:.4e} m')

    rmse_baseline_normalized = _aggregate_normalized_rmse_baseline(rmse_entries, sigma)
    print(f'  precompute: rmse_baseline_normalized = {rmse_baseline_normalized:.4e}')

    # --- segment length diagnostic ---
    print('  precompute: running segment diagnostic')
    from lpv_lfr_baseline.scripts.segment_diag import run_segment_diag  # noqa: PLC0415
    segment_len = run_segment_diag(traj_specs, traj_dir, save_dir)
    print(f'  precompute: chosen segment_len = {segment_len}')

    return {
        'trajs':                    trajs,
        'sigma':                    sigma,
        'rmse_entries':             rmse_entries,
        'rmse_baseline_normalized': rmse_baseline_normalized,
        'segment_len':              segment_len,
    }


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def precompute(traj_specs, traj_dir, save_dir, dtype=torch.float64, force=False):
    """
    Load or compute all fixed precomputed data for parameter recovery training.

    Parameters
    ----------
    traj_specs : sequence of dicts with keys 'id', 'group', 'file'
    traj_dir   : directory containing the .mat trajectory files
    save_dir   : directory for the cache file and segment diagnostic outputs
    dtype      : torch dtype for all output tensors (default float64)
    force      : if True, recompute even when a valid cache exists

    Returns
    -------
    dict with keys:
        trajs, sigma, rmse_entries, rmse_baseline_normalized,
        segment_len, version, fingerprint
    """
    os.makedirs(save_dir, exist_ok=True)
    cache_path = Path(save_dir) / 'precomputed.pt'
    fp = _fingerprint(traj_specs, dtype)

    if not force and cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        if cached.get('fingerprint') == fp:
            print(f'  precompute: loaded from cache ({cache_path})')
            return cached
        print('  precompute: cache fingerprint mismatch — recomputing')

    data = _compute(traj_specs, traj_dir, save_dir, dtype)
    data['version']     = CACHE_VERSION
    data['fingerprint'] = fp
    torch.save(data, cache_path)
    print(f'  precompute: saved to {cache_path}')
    return data


# ----------------------------------------------------------------------
# Verification
# (run as: conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.precompute)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from lpv_lfr_baseline.core.physics import P as _P_test, ts as _ts_test

    dtype = torch.float64

    def check(name, cond, detail=''):
        status = 'PASS' if cond else 'FAIL'
        suffix = f'  ({detail})' if detail else ''
        print(f'  {name:<50s}  {status}{suffix}')
        return cond

    print('=' * 60)
    print('precompute.py structural verification')
    print('=' * 60)

    results = []

    # ------------------------------------------------------------------
    # Check 1 — _build_state_traj_logical: shapes and finite-difference
    # ------------------------------------------------------------------
    print('\nCheck 1: _build_state_traj_logical')
    T = 10
    ts_val = float(_ts_test)
    P_test = _P_test.to(dtype)
    # Build ramp in logical space, then convert to stage so round-trip is exact
    q_logical_ref = torch.zeros(T, 3, dtype=dtype)
    q_logical_ref[:, 2] = torch.linspace(0.3, 0.1, T, dtype=dtype)   # Y ramp
    q_stage = q_logical_ref @ P_test   # q_stage = q_logical @ P  (row convention)

    state = _build_state_traj_logical(q_stage, _P_test, ts_val, dtype)
    results.append(check('output shape (T, 6)', state.shape == (T, 6),
                         f'got {tuple(state.shape)}'))
    results.append(check('dtype preserved', state.dtype == dtype))
    results.append(check('q_logical matches reference',
                         (state[:, :3] - q_logical_ref).abs().max().item() < 1e-12))
    results.append(check('qdot finite', state[:, 3:].isfinite().all().item()))
    # Interior qdot_Y (index 5) must be exactly constant for a linear ramp
    # (central differences on a linear function give exact constant derivative)
    interior_qdot_Y = state[1:-1, 5]
    vel_std = interior_qdot_Y.std().item()
    results.append(check('interior qdot_Y exactly constant (linear ramp)',
                         vel_std < 1e-10, f'std={vel_std:.2e}'))

    # ------------------------------------------------------------------
    # Check 2 — _compute_sigma: clamp and shape
    # ------------------------------------------------------------------
    print('\nCheck 2: _compute_sigma')
    fake_trajs = [
        {'id': 'T1', 'group': 'g1',
         'q1': torch.cat([torch.zeros(100, 2, dtype=dtype),
                          torch.randn(100, 1, dtype=dtype) * 0.05], dim=1)},
        {'id': 'T2', 'group': 'g1',
         'q1': torch.zeros(100, 3, dtype=dtype)},   # all-zero: should clamp
    ]
    sigma = _compute_sigma(fake_trajs, dtype)
    results.append(check('sigma keys match traj ids',
                         set(sigma.keys()) == {'T1', 'T2'}))
    results.append(check('sigma shape (3,)',
                         all(v.shape == (3,) for v in sigma.values())))
    results.append(check('sigma dtype preserved',
                         all(v.dtype == dtype for v in sigma.values())))
    results.append(check('dormant channels clamped to >= 1e-4',
                         sigma['T2'].min().item() >= 1e-4,
                         f'min={sigma["T2"].min().item():.2e}'))
    results.append(check('active channel > 1e-4',
                         sigma['T1'][2].item() > 1e-4,
                         f'sigma_Y={sigma["T1"][2].item():.4e}'))

    # ------------------------------------------------------------------
    # Check 3 — _aggregate_grouped_rmse: group balancing
    # ------------------------------------------------------------------
    print('\nCheck 3: _aggregate_grouped_rmse')
    entries_single = [{'id': 'T1', 'group': 'g1', 'mse_total': 0.04, 'rmse_total': 0.2}]
    entries_two_groups = [
        {'id': 'T1', 'group': 'g1', 'mse_total': 0.01, 'rmse_total': 0.1},
        {'id': 'T2', 'group': 'g1', 'mse_total': 0.09, 'rmse_total': 0.3},
        {'id': 'T3', 'group': 'g2', 'mse_total': 0.25, 'rmse_total': 0.5},
    ]
    rmse_single = _aggregate_grouped_rmse(entries_single)
    results.append(check('single traj: returns rmse_total directly',
                         abs(rmse_single - 0.2) < 1e-12, f'{rmse_single:.4f}'))
    # g1 mean_mse = (0.01+0.09)/2 = 0.05; g2 mean_mse = 0.25; overall = sqrt((0.05+0.25)/2)
    expected = ((0.05 + 0.25) / 2) ** 0.5
    rmse_two = _aggregate_grouped_rmse(entries_two_groups)
    results.append(check('two groups: group-balanced correctly',
                         abs(rmse_two - expected) < 1e-12,
                         f'{rmse_two:.6f} vs {expected:.6f}'))

    # ------------------------------------------------------------------
    # Check 4 — _fingerprint: dtype change invalidates cache
    # ------------------------------------------------------------------
    print('\nCheck 4: _fingerprint cache invalidation')
    specs = [{'id': 'T1', 'file': 'f1.mat', 'group': 'g1'}]
    fp32 = _fingerprint(specs, torch.float32)
    fp64 = _fingerprint(specs, torch.float64)
    specs2 = [{'id': 'T2', 'file': 'f2.mat', 'group': 'g1'}]
    fp_other = _fingerprint(specs2, torch.float64)
    results.append(check('float32 != float64 fingerprint', fp32 != fp64))
    results.append(check('different traj_specs invalidates cache', fp64 != fp_other))
    results.append(check('same inputs produce same fingerprint',
                         _fingerprint(specs, torch.float64) == fp64))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print('=' * 60)
    print(f"Overall: {'ALL PASS' if all(results) else 'SOME FAILED'}")
    print('=' * 60)
    print()
    print('Note: full precompute() requires real .mat trajectory files.')
    print('Run train_param_recovery.py to trigger end-to-end precomputation.')
