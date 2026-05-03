"""
precompute.py
-------------
One-time setup: computes and caches all data that does not depend on trainable
parameters. Writes a single .pt cache file. Subsequent calls return the cached
result immediately.

Computes
--------
trajs                    : list of dicts — per-trajectory u (1,T,3), q1 (T,3),
                           state_traj (T,6), and metadata (id, file, N, fs)
sigma                    : dict traj_id -> (3,) — per-channel output std for
                           loss normalization, clamped to min 1e-4.
                           Mode controlled by norm_mode (see below).
rmse_entries             : list of dicts — per-trajectory detuned-model RMSE
rmse_baseline_normalized : float — simple mean RMSE baseline in sigma units
segment_len              : int — BPTT window length chosen by segment diagnostic
metadata                 : dict — human-readable record of what was computed
                           (dtype, norm_mode, traj_ids, n_timesteps, segment_len)

norm_mode options
-----------------
'per_traj'  (default) : each trajectory normalized by its own per-channel std.
'global'              : single sigma pooled across all trajectories, applied
                        uniformly — same (3,) tensor returned for every traj_id.

All tensors are on CPU in the requested dtype. The training loop moves them to
the training device via .to(device).

What is NOT computed here
-------------------------
G, alpha, beta, gamma, N0, N1, N2 — all depend on trainable parameters and must
be rebuilt each forward pass from current log_params.
"""

import contextlib
import hashlib
import inspect
import io
import os
from pathlib import Path

import torch
from scipy.io import loadmat

from lpv_lfr_baseline.core.physics import P as _P, ts as _ts
from lpv_lfr_baseline.scripts.data_utils import compute_rmse_baseline_metrics


# ----------------------------------------------------------------------
# Cache helpers
# ----------------------------------------------------------------------

def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _fingerprint(traj_specs, traj_dir, dtype, norm_mode, overlap_fraction=0.0, D=None):
    """Cache key — invalidated by data changes (mtimes), logic changes (compute_hash), or config.

    D is included when known (computed from data via _diag_fft). A change in D due to updated
    _diag_fft logic invalidates the cache even if data mtimes are unchanged.
    """
    fp = {
        'traj_specs':       tuple((s['id'], s['file']) for s in traj_specs),
        'mtimes':           tuple((s['file'], _mtime(os.path.join(traj_dir, s['file']))) for s in traj_specs),
        'dtype':            str(dtype),
        'norm_mode':        norm_mode,
        'overlap_fraction': overlap_fraction,
        'compute_hash':     _COMPUTE_HASH,
    }
    if D is not None:
        fp['D'] = D
    return fp


# ----------------------------------------------------------------------
# Decimation factor helper (used for fingerprinting)
# ----------------------------------------------------------------------

def _compute_D_from_specs(traj_specs, traj_dir, dtype):
    """Load trajectories and run FFT diagnostic (silently) to obtain decimation factor D.

    Called once per precompute() invocation so D can be included in the cache fingerprint.
    Output is suppressed; no plots are saved.
    """
    from lpv_lfr_baseline.scripts.experiment_diagnostics import _diag_fft  # noqa: PLC0415
    trajs = []
    for spec in traj_specs:
        mat_path = os.path.join(traj_dir, spec['file'])
        u, q1, fs = _load_trajectory(mat_path, dtype)
        trajs.append({'id': spec['id'], 'N': int(q1.shape[0]), 'fs': fs, 'u': u, 'q1': q1})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = _diag_fft(trajs, save_dir=None)
    return r['decimation_factor']


# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------

def _load_trajectory(mat_path, dtype):
    """Load one .mat trajectory. Returns (u, q1, fs).

    u is the total stage-coordinate force applied to the plant: u_q1 + f_sim.
    For non-multisine trajectories f_sim is all zeros, so this reduces to u_q1.
    For multisine trajectories f_sim is the feedforward force; omitting it would
    feed only the feedback force (~800 N RMS) to the model while q1 was produced
    by the net force (~40-60 N RMS), causing a large systematic input mismatch.
    """
    mat = loadmat(mat_path)
    u_q1  = mat['u_q1']                                            # (T, 3) stage coords
    f_sim = mat.get('f_sim', 0)                                    # (T, 3) or 0; absent in ref_injection (always zero there)
    u     = torch.tensor(u_q1 + f_sim, dtype=dtype).unsqueeze(0)  # (1, T, 3) total plant force
    q1    = torch.tensor(mat['q1'],     dtype=dtype)               # (T, 3)
    fs    = float(mat['fs'].squeeze()) if 'fs' in mat else None
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
    # Use torch.linalg.solve to compute numerical instead of polynomial expression
    q_logical = torch.linalg.solve(P.to(dtype).T, q1_stage.to(dtype).T).T   # (T, 3)

    qdot = torch.empty_like(q_logical)
    if q_logical.shape[0] == 1:
        qdot.zero_()
    else:
        qdot[0]    = (q_logical[1]  - q_logical[0])  / ts_val
        qdot[1:-1] = (q_logical[2:] - q_logical[:-2]) / (2 * ts_val)
        qdot[-1]   = (q_logical[-1] - q_logical[-2])  / ts_val

    return torch.cat([q_logical, qdot], dim=-1)   # (T, 6)


def _compute_sigma(trajs, norm_mode, dtype):
    """
    Compute sigma for loss normalization. All values clamped to min 1e-4.

    norm_mode='per_traj' : per-trajectory per-channel std. Channels with
                           near-zero variance are clamped so they produce a
                           finite but small loss signal.
    norm_mode='global'   : single sigma pooled across all trajectories,
                           returned as a dict with the same tensor for every id.
    """
    if norm_mode == 'global':
        q1_all = torch.cat([traj['q1'] for traj in trajs], dim=0)   # (T_total, 3)
        global_sigma = torch.empty(3, dtype=dtype)
        for c in range(3):
            global_sigma[c] = q1_all[:, c].std().clamp(min=1e-4)
        return {traj['id']: global_sigma for traj in trajs}
    if norm_mode != 'per_traj':
        raise ValueError(f"norm_mode must be 'per_traj' or 'global', got {norm_mode!r}")
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

def _aggregate_rmse(entries):
    """Simple mean RMSE across all trajectories."""
    return (sum(e['mse_total'] for e in entries) / len(entries)) ** 0.5


def _aggregate_normalized_rmse_baseline(rmse_entries, sigma):
    """
    Simple mean RMSE baseline in sigma-normalized (dimensionless) units.

    Uses float64 throughout for the scalar computation regardless of training
    dtype — normalization constants should be computed at full precision.
    """
    def _normalized_mse(entry):
        rmse_ch = torch.tensor(entry['rmse_ch'], dtype=torch.float64)
        s = sigma[entry['id']].to(torch.float64)
        return (rmse_ch / s).pow(2).mean().item()

    overall_mse = sum(_normalized_mse(e) for e in rmse_entries) / len(rmse_entries)
    return overall_mse ** 0.5


# ----------------------------------------------------------------------
# Segment pool helpers
# ----------------------------------------------------------------------

def _build_segment_pools(trajs, segment_len, overlap_fraction=0.0):
    """
    Pre-compute valid segment start indices for each trajectory.

    stride = segment_len * (1 - overlap_fraction), rounded to nearest sample.
    Returns dict traj_id -> list[int] of start indices.
    Raises ValueError if any trajectory is shorter than segment_len.
    """
    stride = max(1, round(segment_len * (1 - overlap_fraction)))
    pools  = {}
    for traj in trajs:
        T      = traj['N']
        starts = list(range(0, T - segment_len + 1, stride))
        if not starts:
            raise ValueError(
                f'Trajectory {traj["id"]} (N={T}) is shorter than segment_len={segment_len}. '
                f'Reduce segment_len or use a longer trajectory.'
            )
        pools[traj['id']] = starts
    return pools


# ----------------------------------------------------------------------
# Core computation
# ----------------------------------------------------------------------

def _compute(traj_specs, traj_dir, save_dir, dtype, norm_mode, overlap_fraction):
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
            'file':       spec['file'],
            'N':          int(q1.shape[0]),
            'fs':         fs,
            'u':          u,            # (1, T, 3) CPU
            'q1':         q1,           # (T, 3)    CPU
            'state_traj': state_traj,   # (T, 6)    CPU
        })
        print(f'    {spec["id"]}: T={q1.shape[0]}, fs={fs} Hz')

    # --- FFT diagnostic: determine decimation factor and new sampling rate ---
    print('  precompute: running FFT diagnostic')
    from lpv_lfr_baseline.scripts.experiment_diagnostics import (  # noqa: PLC0415
        recommend_segment_len, _diag_fft,
    )
    fs_orig = float(trajs[0]['fs'])
    r_fft   = _diag_fft(trajs, save_dir)
    D       = r_fft['decimation_factor']
    fs_new  = float(r_fft['fs_new'])

    # --- decimate all trajectories in-place ---
    print(f'  precompute: decimating trajectories by D={D}  ({fs_orig:.0f} -> {fs_new:.0f} Hz)')
    for traj in trajs:
        traj['u']          = traj['u'][:, ::D, :]     # (1, T//D, 3)
        traj['q1']         = traj['q1'][::D, :]        # (T//D, 3)
        traj['state_traj'] = traj['state_traj'][::D, :] # (T//D, 6)
        traj['N']          = traj['u'].shape[1]
        traj['fs']         = fs_new
        print(f'    {traj["id"]}: T_dec={traj["N"]}')

    ts_eff = float(_ts) * D   # effective timestep after decimation [s]

    # --- sigma (on decimated data) ---
    print(f'  precompute: computing sigma  (norm_mode={norm_mode!r})')
    sigma = _compute_sigma(trajs, norm_mode, dtype)
    for traj in trajs:
        s = sigma[traj['id']]
        print(f'    {traj["id"]}: sigma = [{s[0]:.3e}, {s[1]:.3e}, {s[2]:.3e}]')

    # --- RMSE baseline (detuned initial model, runs at original resolution from .mat) ---
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
            'file':       traj['file'],
            'mse_total':  metrics['mse_total'],
            'rmse_total': metrics['rmse_total'],
            'rmse_ch':    metrics['rmse_ch'],
        })
        print(f'    {traj["id"]}: RMSE = {metrics["rmse_total"]:.4e} m')

    rmse_baseline_normalized = _aggregate_normalized_rmse_baseline(rmse_entries, sigma)
    print(f'  precompute: rmse_baseline_normalized = {rmse_baseline_normalized:.4e}')

    # --- segment length diagnostic (based on oscillatory poles at fs_new) ---
    print('  precompute: running segment length diagnostic')
    segment_len = recommend_segment_len(fs_orig, fs_new, save_dir, dtype=dtype)
    print(f'  precompute: chosen segment_len = {segment_len}  (at {fs_new:.0f} Hz)')

    # --- segment pools ---
    pools = _build_segment_pools(trajs, segment_len, overlap_fraction)
    print(f'  precompute: pool sizes = { {tid: len(s) for tid, s in pools.items()} }')

    metadata = {
        'traj_dir':         traj_dir,
        'dtype':            str(dtype),
        'norm_mode':        norm_mode,
        'overlap_fraction': overlap_fraction,
        'traj_ids':         [t['id'] for t in trajs],
        'traj_files':       [t['file'] for t in trajs],
        'n_timesteps':      {t['id']: t['N'] for t in trajs},
        'segment_len':      segment_len,
        'D':                D,
        'fs_new':           fs_new,
        'ts_eff':           ts_eff,
    }

    return {
        'trajs':                    trajs,
        'sigma':                    sigma,
        'rmse_entries':             rmse_entries,
        'rmse_baseline_normalized': rmse_baseline_normalized,
        'segment_len':              segment_len,
        'pools':                    pools,
        'metadata':                 metadata,
        'D':                        D,
        'fs_new':                   fs_new,
        'ts_eff':                   ts_eff,
    }


_COMPUTE_HASH = hashlib.md5(inspect.getsource(_compute).encode()).hexdigest()[:8]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def precompute(traj_specs, traj_dir, save_dir, dtype=torch.float64,
               norm_mode='per_traj', overlap_fraction=0.0, force=False):
    """
    Load or compute all fixed precomputed data for parameter recovery training.

    Parameters
    ----------
    traj_specs       : sequence of dicts with keys 'id', 'file'
    traj_dir         : directory containing the .mat trajectory files
    save_dir         : directory for the cache file and segment diagnostic outputs
    dtype            : torch dtype for all output tensors (default float64)
    norm_mode        : 'per_traj' (default) or 'global' — controls how sigma is computed.
                       Changing this invalidates the cache and forces a full recompute.
    overlap_fraction : fraction of segment_len used as overlap between consecutive segments.
                       Changing this only rebuilds pools — no full recompute needed.
    force            : if True, recompute even when a valid cache exists

    Returns
    -------
    dict with keys:
        trajs, sigma, rmse_entries, rmse_baseline_normalized,
        segment_len, pools, metadata, fingerprint
    """
    os.makedirs(save_dir, exist_ok=True)
    cache_path = Path(save_dir) / 'precomputed.pt'
    # Compute D from data so it can be included in the fingerprint as a cache key.
    # A change in D (e.g., from updated _diag_fft logic) will therefore invalidate the cache.
    D = _compute_D_from_specs(traj_specs, traj_dir, dtype)
    fp = _fingerprint(traj_specs, traj_dir, dtype, norm_mode, overlap_fraction, D=D)

    if not force and cache_path.exists():
        cached    = torch.load(cache_path, weights_only=False)
        cached_fp = cached.get('fingerprint', {})
        if cached_fp == fp:
            print(f'  precompute: loaded from cache ({cache_path})')
            meta = cached.get('metadata', {})
            print(f'    traj_dir={meta.get("traj_dir")}  norm_mode={meta.get("norm_mode")!r}  '
                  f'trajs={meta.get("traj_ids")}  segment_len={meta.get("segment_len")}  '
                  f'overlap={meta.get("overlap_fraction", 0.0):.0%}')
            return cached

        # Fast path: only overlap_fraction changed — rebuild pools without full recompute
        fp_core        = {k: v for k, v in fp.items()        if k != 'overlap_fraction'}
        cached_fp_core = {k: v for k, v in cached_fp.items() if k != 'overlap_fraction'}
        if fp_core == cached_fp_core and 'segment_len' in cached:
            print(f'  precompute: overlap_fraction changed — rebuilding pools only')
            cached['pools']                       = _build_segment_pools(
                cached['trajs'], cached['segment_len'], overlap_fraction
            )
            cached['fingerprint']['overlap_fraction']  = overlap_fraction
            cached['metadata']['overlap_fraction']     = overlap_fraction
            torch.save(cached, cache_path)
            print(f'  precompute: pools updated and saved to {cache_path}')
            return cached

        print('  precompute: cache stale — recomputing. Changed:')
        for key, val in fp.items():
            old = cached_fp.get(key)
            if old != val:
                if key == 'mtimes':
                    old_d = dict(old or [])
                    for fname, t in val:
                        if old_d.get(fname) != t:
                            print(f'    mtime changed: {fname}')
                else:
                    print(f'    {key}: {old!r} → {val!r}')

    data = _compute(traj_specs, traj_dir, save_dir, dtype, norm_mode, overlap_fraction)
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
        {'id': 'T1',
         'q1': torch.cat([torch.zeros(100, 2, dtype=dtype),
                          torch.randn(100, 1, dtype=dtype) * 0.05], dim=1)},
        {'id': 'T2',
         'q1': torch.zeros(100, 3, dtype=dtype)},   # all-zero: should clamp
    ]
    sigma = _compute_sigma(fake_trajs, 'per_traj', dtype)
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
    sigma_g = _compute_sigma(fake_trajs, 'global', dtype)
    results.append(check('global: all traj_ids return same tensor object',
                         sigma_g['T1'] is sigma_g['T2']))
    results.append(check('global: shape (3,)', sigma_g['T1'].shape == (3,)))

    # ------------------------------------------------------------------
    # Check 3 — _aggregate_rmse: simple mean
    # ------------------------------------------------------------------
    print('\nCheck 3: _aggregate_rmse')
    entries_single = [{'id': 'T1', 'mse_total': 0.04, 'rmse_total': 0.2}]
    entries_multi = [
        {'id': 'T1', 'mse_total': 0.01, 'rmse_total': 0.1},
        {'id': 'T2', 'mse_total': 0.09, 'rmse_total': 0.3},
        {'id': 'T3', 'mse_total': 0.25, 'rmse_total': 0.5},
    ]
    rmse_single = _aggregate_rmse(entries_single)
    results.append(check('single traj: sqrt(mse_total)',
                         abs(rmse_single - 0.2) < 1e-12, f'{rmse_single:.4f}'))
    # simple mean: sqrt((0.01 + 0.09 + 0.25) / 3)
    expected = ((0.01 + 0.09 + 0.25) / 3) ** 0.5
    rmse_multi = _aggregate_rmse(entries_multi)
    results.append(check('multi-traj: simple mean RMSE',
                         abs(rmse_multi - expected) < 1e-12,
                         f'{rmse_multi:.6f} vs {expected:.6f}'))

    # ------------------------------------------------------------------
    # Check 4 — _fingerprint: dtype change invalidates cache
    # ------------------------------------------------------------------
    print('\nCheck 4: _fingerprint cache invalidation')
    specs     = [{'id': 'T1', 'file': 'f1.mat'}]
    fake_dir  = os.getcwd()
    fp32      = _fingerprint(specs, fake_dir, torch.float32, 'per_traj')
    fp64      = _fingerprint(specs, fake_dir, torch.float64, 'per_traj')
    fp_global = _fingerprint(specs, fake_dir, torch.float64, 'global')
    specs2    = [{'id': 'T2', 'file': 'f2.mat'}]
    fp_other  = _fingerprint(specs2, fake_dir, torch.float64, 'per_traj')
    results.append(check('float32 != float64 fingerprint', fp32 != fp64))
    results.append(check('norm_mode change invalidates cache', fp64 != fp_global))
    results.append(check('different traj_specs invalidates cache', fp64 != fp_other))
    results.append(check('same inputs produce same fingerprint',
                         _fingerprint(specs, fake_dir, torch.float64, 'per_traj') == fp64))
    results.append(check('compute_hash present in fingerprint', 'compute_hash' in fp64))

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
