"""
train_param_recovery.py
-----------------------
Step 3b: recover true physical parameters from MATLAB data using batched multiple
shooting.

Approach:
    Full state trajectories (positions + finite-difference velocities) are cached
    once per source trajectory in logical coordinates.
    At startup, active parameter-recovery trajectories are load-or-build cached.
    Each epoch samples a balanced segment batch across the active trajectory groups.
    Segment start states come from cached state_traj; u_seg and q1_seg are sliced
    directly from the selected trajectories.
    Loss: MSE(Y_pred, q1_seg) + block.param_loss()
    Optimizer: Adam on block.log_params only.

Training data: Matlab-output/parameter-recovery/*.mat
    Active trajectories are selected from T1-T6 below. Using one active trajectory
    is effectively single-trajectory training; using several is multi-trajectory
    training. The code path is the same in both cases.

Run as:
    conda run -n GraduationProject python -m lpv_lfr_baseline.train_param_recovery
"""

import contextlib
import os
import sys
import time
import threading
import queue

import torch
import torch.nn.functional as F
import torch.profiler
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lpv_lfr_baseline.blocks.lfr_param_block import (
    ParameterizedLFRBlock,
    _PARAM_NAMES,
    _TRUE_PARAMS,
    _build_matrices,
)
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.lfr_simulate import simulate
from lpv_lfr_baseline.core.physics import P as _P_PHYSICS, ts as _TS_PHYSICS, build_poly_constants
from lpv_lfr_baseline.scripts.data_utils import compute_rmse_baseline_metrics

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TRAJ_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'Matlab-output', 'parameter-recovery'
)
SAVE_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'simulations', 'param_recovery'
)

TRAJ_SPECS = (
    {'id': 'T1', 'group': 'y_only', 'file': 'T1_Y_sweep_conservative.mat'},
    {'id': 'T6', 'group': 'y_only', 'file': 'T6_Y_sweep_aggressive.mat'},
    {'id': 'T2', 'group': 'x_sym_mh', 'file': 'T2_X_sym_Y030.mat'},
    {'id': 'T3', 'group': 'x_sym_mh', 'file': 'T3_X_sym_Y000.mat'},
    {'id': 'T4', 'group': 'rot_coupled', 'file': 'T4_X_antisym_Y020.mat'},
    {'id': 'T5', 'group': 'rot_coupled', 'file': 'T5_X_sym_Y_sweep.mat'},
)
TRAJ_GROUPS = tuple(dict.fromkeys(spec['group'] for spec in TRAJ_SPECS))  # canonical group order derived from TRAJ_SPECS
ACTIVE_TRAJ_IDS = tuple(spec['id'] for spec in TRAJ_SPECS)

# Binary channel masks per trajectory — channels [X1, X2, Y].
# A 0 means the channel is dormant (controller-suppressed, no informative residual)
# and must not contribute gradient signal.  See D-044 and docs/loss-function-design.md.
CHANNEL_MASKS = {
    'T1': [0, 0, 1],   # y_only: only Y is excited
    'T6': [0, 0, 1],   # y_only: only Y is excited
    'T2': [1, 1, 0],   # x_sym_mh: X1 and X2 excited, Y held fixed
    'T3': [1, 1, 0],   # x_sym_mh: X1 and X2 excited, Y held fixed
    'T4': [1, 1, 0],   # rot_coupled: X1 and X2 excited (antisym), Y held fixed
    'T5': [1, 1, 1],   # rot_coupled+Y: all three channels active
}

# ── Experiment settings ───────────────────────────────────────────────────────
N_STEPS = None  # cap on steps (None = use all); overridden to 500 when PROFILE=True
EPOCHS = 3
LR = 1e-3
FULL_EVAL_INTERVAL = 10   # run full-trajectory eval every N epochs
SEGMENT_LEN = None  # None = choose the smallest stable candidate from the segment-length diagnostic; int = use that fixed number of samples
PARAM_LOSS_WEIGHT = 0.0
SPLIT_REG_WEIGHT = 1e-2   # D-037: scale-invariant penalty on degenerate splits (kb1/kb2, cb1/cb2, Jb/Jh)
LOG_INTERVAL = 25
CHECKPOINT_INTERVAL = 100
PROFILE = True
TIME_EPOCHS = False
TRAIN_SEGMENTS_PER_EPOCH = 8
VAL_SEGMENTS_FIXED = 8

# ── Internal / cache constants ────────────────────────────────────────────────
SEGMENT_DIAG_CANDIDATES_S = (0.1, 0.2, 0.4, 0.6)
SEGMENT_DIAG_SEGMENTS_PER_GROUP = 8
BASE_SEED = 1234
SEGMENT_DIAG_VERSION = 1
RMSE_BASELINE_CACHE_VERSION = 1
SIGMA_CACHE_VERSION = 2  # v2: per-trajectory per-channel dict (was global (3,) tensor in v1)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _load_trajectory(mat_path):
    """Load one MATLAB trajectory as float64 tensors."""
    mat = loadmat(mat_path)
    u = torch.tensor(mat['u_q1'], dtype=torch.float64).unsqueeze(0)  # (1, N, 3)
    q1 = torch.tensor(mat['q1'], dtype=torch.float64)  # (N, 3)
    fs = float(mat['fs'].squeeze()) if 'fs' in mat else None
    return u, q1, fs


def _active_traj_specs():
    """Return the active trajectory specs in library order."""
    active_ids = set(ACTIVE_TRAJ_IDS)
    specs = tuple(spec for spec in TRAJ_SPECS if spec['id'] in active_ids)
    missing = active_ids.difference(spec['id'] for spec in specs)
    if missing:
        raise ValueError(f'Unknown ACTIVE_TRAJ_IDS: {sorted(missing)}')
    if not specs:
        raise ValueError('ACTIVE_TRAJ_IDS must contain at least one trajectory id')
    return specs


def _active_groups_from_specs(traj_specs):
    """Return active groups in the canonical group order."""
    return tuple(group for group in TRAJ_GROUPS if any(spec['group'] == group for spec in traj_specs))


def _active_groups_from_trajs(trajs):
    """Return active groups for already loaded trajectories."""
    groups = {traj['group'] for traj in trajs}
    return tuple(group for group in TRAJ_GROUPS if group in groups)


def _traj_set_tag(traj_specs):
    """Stable tag for cache/save files derived from the active trajectory ids."""
    return '-'.join(spec['id'] for spec in traj_specs)


def _balanced_group_counts(total_segments, active_groups):
    """Distribute a total segment budget as evenly as possible across groups."""
    n_groups = len(active_groups)
    base = total_segments // n_groups
    rem = total_segments % n_groups
    return {
        group: base + (1 if i < rem else 0)
        for i, group in enumerate(active_groups)
    }


def _run_no_grad(block, x0, u):
    """Simulate full trajectory with current params, no gradient. Used for evaluation."""
    with torch.no_grad():
        params = block._recover_params()
        kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh = params
        _, M1, M2, K, C = _build_matrices(
            torch.stack([kb1 + kb2, cg1, cg2, cy, cb1 + cb2, mh, m1, m2, mb, Jb + Jh]),
            block._Lb,
            block._d,
        )
        alpha, beta, gamma, N0, N1, N2 = build_poly_constants(
            m1, m2, mb, mh, Jb, Jh, block._Lb, block._d
        )
        d0 = mh * (alpha * gamma - beta ** 2)
        G = build_G_matrix(N0, d0, M1, M2, K, C)
        return simulate(
            x0,
            u,
            G,
            K,
            C,
            mh,
            alpha,
            beta,
            gamma,
            N0,
            N1,
            N2,
            block._P,
            block._ts,
            bptt_mode='full',
            return_latents=False,
        )


def _full_traj_eval(block, trajs):
    """Full-trajectory eval (no grad). Same metric as Step 5. Returns (rmse_m, entries)."""
    entries = []
    for traj in trajs:
        pre = _run_no_grad(block, traj['state_traj'][:1], traj['u'])
        diff = pre.Y[0] - traj['q1']
        rmse = diff.pow(2).mean().item() ** 0.5
        entries.append({
            'id': traj['id'], 'group': traj['group'],
            'mse_total': rmse ** 2, 'rmse_total': rmse,
            'rmse_ch': diff.pow(2).mean(dim=0).sqrt().cpu(),
        })
    return _aggregate_grouped_rmse(entries), entries


def _state_cache_path(save_dir, traj_id, n_steps):
    """Stable cache filename per trajectory and trajectory length."""
    return os.path.join(save_dir, f'state_traj_{traj_id}_n{n_steps}.pt')


def _build_state_traj_logical(q1_stage, P, ts):
    """Build [q, qdot] in logical coordinates from stage-position data."""
    q_stage = q1_stage.cpu()
    q_logical = torch.linalg.solve(P.detach().cpu().T, q_stage.T).T
    ts_val = float(ts)

    qdot_logical = torch.empty_like(q_logical)
    if q_logical.shape[0] == 1:
        qdot_logical.zero_()
    else:
        qdot_logical[0] = (q_logical[1] - q_logical[0]) / ts_val
        qdot_logical[1:-1] = (q_logical[2:] - q_logical[:-2]) / (2 * ts_val)
        qdot_logical[-1] = (q_logical[-1] - q_logical[-2]) / ts_val

    return torch.cat([q_logical, qdot_logical], dim=-1)


def _load_or_build_state_traj(traj_id, q1_stage, P, ts, device, save_dir, load=True):
    """
    Load cached logical state trajectory if present, otherwise compute once.

    When load=False, only the cache existence is ensured; the tensor is not kept
    resident in memory.
    """
    cache_path = _state_cache_path(save_dir, traj_id, q1_stage.shape[0])
    if os.path.exists(cache_path):
        print(f'  state_traj[{traj_id}]: loaded from cache')
        return torch.load(cache_path, map_location=device) if load else None

    traj = _build_state_traj_logical(q1_stage, P, ts)
    torch.save(traj, cache_path)
    print(f'  state_traj[{traj_id}]: computed and cached')
    return traj.to(device) if load else None


def _load_grouped_trajectories(traj_specs, traj_dir, P, ts, device, save_dir, load_tensors, n_steps=None):
    """Load grouped trajectories and ensure per-trajectory state caches exist."""
    trajs = []
    grouped = {group: [] for group in _active_groups_from_specs(traj_specs)}
    for spec in traj_specs:
        mat_path = os.path.join(traj_dir, spec['file'])
        u_i, q1_i, fs_i = _load_trajectory(mat_path)
        state_traj_i = _load_or_build_state_traj(
            spec['id'], q1_i, P, ts, device, save_dir, load=load_tensors
        )
        if load_tensors and n_steps is not None:
            u_i = u_i[:, :n_steps, :]
            q1_i = q1_i[:n_steps]
            state_traj_i = state_traj_i[:n_steps]
        label = f"{spec['id']} ({q1_i.shape[0] / fs_i:.2f}s)" if fs_i else spec['id']
        grouped[spec['group']].append(label)
        traj = {
            'id': spec['id'],
            'group': spec['group'],
            'file': spec['file'],
            'N': int(q1_i.shape[0]),
            'fs': fs_i,
        }
        if load_tensors:
            traj['u'] = u_i.to(device)
            traj['q1'] = q1_i.to(device)
            traj['state_traj'] = state_traj_i
        trajs.append(traj)

    for group, items in grouped.items():
        print(f"  {group:<12}: {', '.join(items)}")
    return trajs


def _rmse_baseline_cache_path(save_dir):
    """Cache path for per-trajectory RMSE_baseline metrics."""
    return os.path.join(save_dir, f'rmse_baseline_cache_v{RMSE_BASELINE_CACHE_VERSION}.pt')


def _load_rmse_baseline_cache(save_dir):
    """Load the RMSE_baseline cache or create an empty one."""
    cache_path = _rmse_baseline_cache_path(save_dir)
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu')
        if cached.get('version') == RMSE_BASELINE_CACHE_VERSION:
            return cached
    return {'version': RMSE_BASELINE_CACHE_VERSION, 'per_traj': {}}


def _aggregate_grouped_rmse(selected_entries):
    """Aggregate per-trajectory MSE values into one group-balanced RMSE scalar."""
    if len(selected_entries) == 1:
        return float(selected_entries[0]['rmse_total'])

    group_mse = {}
    for entry in selected_entries:
        group_mse.setdefault(entry['group'], []).append(entry['mse_total'])

    overall_mse = sum(sum(values) / len(values) for values in group_mse.values()) / len(group_mse)
    return overall_mse ** 0.5


def _print_rmse_baseline_summary(selected_entries, overall_rmse):
    """Print the cached/computed RMSE_baseline values relevant to the current run."""
    if len(selected_entries) == 1:
        entry = selected_entries[0]
        print(f"  {entry['id']}: RMSE = {entry['rmse_total']:.6e} m")
    else:
        print(f'  {"Traj":<6}  {"Group":<12}  {"RMSE [m]":>12}')
        print(f'  {"-" * 6}  {"-" * 12}  {"-" * 12}')
        for entry in selected_entries:
            print(
                f"  {entry['id']:<6}  {entry['group']:<12}  "
                f"{entry['rmse_total']:>12.4e}"
            )
    print(f'  Overall RMSE_baseline = {overall_rmse:.6e} m')


def _get_or_compute_rmse_baseline(traj_specs, device, save_dir):
    """Load cached per-trajectory RMSE_baseline metrics and derive the run scalar."""
    cache = _load_rmse_baseline_cache(save_dir)
    updated = False
    selected_entries = []

    for spec in traj_specs:
        spec_path = os.path.join(TRAJ_DIR, spec['file'])
        entry = cache['per_traj'].get(spec['id'])
        if entry is None or entry.get('file') != spec['file'] or entry.get('group') != spec['group']:
            _, q1_i, _ = _load_trajectory(spec_path)
            state_traj_i = _load_or_build_state_traj(
                spec['id'], q1_i, _P_PHYSICS, _TS_PHYSICS, device, save_dir, load=True
            )
            metrics = compute_rmse_baseline_metrics(
                mat_path=spec_path,
                x0_logical=state_traj_i[:1].cpu(),
                verbose=False,
            )
            entry = {
                'id': spec['id'],
                'file': spec['file'],
                'group': spec['group'],
                **metrics,
            }
            cache['per_traj'][spec['id']] = entry
            updated = True
        selected_entries.append(entry)

    if updated:
        torch.save(cache, _rmse_baseline_cache_path(save_dir))
        print('  RMSE_baseline cache: updated')
    else:
        print('  RMSE_baseline cache: loaded')

    overall_rmse = _aggregate_grouped_rmse(selected_entries)
    _print_rmse_baseline_summary(selected_entries, overall_rmse)
    return overall_rmse, selected_entries


def _sigma_cache_path(save_dir):
    """Cache path for the channel normalisation sigma."""
    return os.path.join(save_dir, f'sigma_v{SIGMA_CACHE_VERSION}.pt')


def _get_or_compute_sigma(save_dir):
    """
    Compute per-trajectory per-channel std and cache the result.

    For each trajectory, sigma[traj_id][c] is the std of channel c computed
    *only* from the time steps where that channel is active (mask == 1).
    Dormant channels (mask == 0) get sigma = 1.0 so that masked residuals
    produce a zero numerator regardless of their denominator value.

    Returns a dict {traj_id: (3,) float64 CPU tensor}.
    See D-044 and docs/loss-function-design.md for the full design rationale.
    """
    cache_path = _sigma_cache_path(save_dir)
    fingerprint = tuple((s['id'], s['file']) for s in TRAJ_SPECS)
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        if cached.get('version') == SIGMA_CACHE_VERSION and cached.get('fingerprint') == fingerprint:
            print('  sigma: loaded from cache')
            return cached['sigma']

    sigma = {}
    for spec in TRAJ_SPECS:
        traj_id = spec['id']
        q1 = _load_trajectory(os.path.join(TRAJ_DIR, spec['file']))[1]  # (N, 3) CPU
        mask = CHANNEL_MASKS[traj_id]
        ch_sigma = torch.ones(3, dtype=torch.float64)
        for c in range(3):
            if mask[c] == 1:
                ch_sigma[c] = q1[:, c].std().clamp(min=1e-4)
            # dormant channels keep sigma = 1.0 (masked residual is 0 anyway)
        sigma[traj_id] = ch_sigma

    torch.save({'version': SIGMA_CACHE_VERSION, 'fingerprint': fingerprint, 'sigma': sigma}, cache_path)
    print('  sigma: computed and cached')
    return sigma


def _aggregate_normalized_rmse_baseline(selected_entries, sigma_dict):
    """
    Group-balanced RMSE_baseline in dimensionless (normalized) units.

    Mirrors _aggregate_grouped_rmse but normalizes per-channel RMSE by the
    per-trajectory per-channel sigma, then averages only over active channels
    (mask == 1) so dormant channels do not contribute.

    sigma_dict is a {traj_id: (3,) CPU tensor} dict (from _get_or_compute_sigma).
    """
    def _entry_mse_norm(entry):
        traj_id = entry['id']
        rmse_ch = torch.tensor(entry['rmse_ch'], dtype=torch.float64)
        sigma = sigma_dict[traj_id]
        mask = torch.tensor(CHANNEL_MASKS[traj_id], dtype=torch.float64)
        n_active = mask.sum()
        if n_active == 0:
            return 0.0
        # Only include active channels in the mean
        normalized_sq = ((rmse_ch / sigma) * mask).pow(2).sum() / n_active
        return normalized_sq.item()

    if len(selected_entries) == 1:
        return _entry_mse_norm(selected_entries[0]) ** 0.5

    group_mse = {}
    for entry in selected_entries:
        group_mse.setdefault(entry['group'], []).append(_entry_mse_norm(entry))
    overall_mse = sum(sum(v) / len(v) for v in group_mse.values()) / len(group_mse)
    return overall_mse ** 0.5


def _attach_valid_start_idx(trajs, segment_len):
    """Attach valid segment starts 0 .. N - segment_len for each trajectory."""
    for traj in trajs:
        n_valid = max(traj['N'] - segment_len + 1, 0)
        traj['valid_start_idx'] = torch.arange(n_valid, dtype=torch.int64)


def _sample_balanced_segments(trajs, segment_len, group_counts, seed):
    """Sample balanced segments according to the provided per-group counts."""
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)

    x0_seg_list = []
    u_seg_list = []
    q1_seg_list = []
    sample_plan = []

    for group, n_per_group in group_counts.items():
        if n_per_group <= 0:
            continue
        group_trajs = [traj for traj in trajs if traj['group'] == group and traj['valid_start_idx'].numel() > 0]
        if not group_trajs:
            raise ValueError(f'No valid trajectories for group {group!r} at segment_len={segment_len}')

        for _ in range(n_per_group):
            traj = group_trajs[torch.randint(len(group_trajs), (1,), generator=generator).item()]
            start_idx = int(
                traj['valid_start_idx'][
                    torch.randint(traj['valid_start_idx'].numel(), (1,), generator=generator).item()
                ].item()
            )
            stop_idx = start_idx + segment_len
            x0_seg_list.append(traj['state_traj'][start_idx])
            u_seg_list.append(traj['u'][0, start_idx:stop_idx, :])
            q1_seg_list.append(traj['q1'][start_idx:stop_idx, :])
            sample_plan.append({
                'traj_id': traj['id'],
                'group': group,
                'start_idx': start_idx,
            })

    return (
        torch.stack(x0_seg_list, dim=0),
        torch.stack(u_seg_list, dim=0),
        torch.stack(q1_seg_list, dim=0),
        sample_plan,
    )


def _build_segment_diag_param_sets():
    """Fixed parameter sets for segment-length discrimination testing."""
    params_true = torch.tensor([_TRUE_PARAMS[name] for name in _PARAM_NAMES], dtype=torch.float64)
    idx = {name: i for i, name in enumerate(_PARAM_NAMES)}
    scales = torch.ones_like(params_true)
    for name in ('kb1', 'kb2', 'cg1', 'cg2'):
        scales[idx[name]] = 1.1
    for name in ('mh', 'Jb', 'Jh'):
        scales[idx[name]] = 0.9

    return (
        {'name': 'true', 'params': params_true},
        {'name': 'all_up_10', 'params': params_true * 1.1},
        {'name': 'all_down_10', 'params': params_true * 0.9},
        {'name': 'coupling_mix', 'params': params_true * scales},
    )


def _set_block_physical_params(block, params_vec):
    """Overwrite block parameters using physical values in _PARAM_NAMES order."""
    with torch.no_grad():
        block.log_params.copy_(params_vec.to(block.log_params.device, dtype=block.log_params.dtype).log())


def _eval_param_set_on_segments(block, wrapper, params_vec, x0_seg, u_seg, q1_seg):
    """Return per-segment MSE for one fixed physical-parameter vector."""
    _set_block_physical_params(block, params_vec)
    with torch.no_grad():
        y_pred = wrapper(x0_seg, u_seg)
    return (y_pred - q1_seg).pow(2).mean(dim=(1, 2)).cpu()


def _segment_diag_cache_path(save_dir, traj_specs):
    """Cache path for the segment-length diagnostic result."""
    return os.path.join(save_dir, f'segment_len_diag_{_traj_set_tag(traj_specs)}_v{SEGMENT_DIAG_VERSION}.pt')


def _segment_diag_cache_matches(cached, traj_specs, param_set_names):
    """Return True when the cached diagnostic matches the current config."""
    return (
        cached.get('version') == SEGMENT_DIAG_VERSION
        and tuple(cached.get('traj_specs', ())) == tuple(traj_specs)
        and tuple(cached.get('candidate_lengths_s', ())) == SEGMENT_DIAG_CANDIDATES_S
        and cached.get('n_per_group') == SEGMENT_DIAG_SEGMENTS_PER_GROUP
        and cached.get('seed') == BASE_SEED
        and tuple(cached.get('param_set_names', ())) == tuple(param_set_names)
    )


def _choose_segment_len_from_diag(results):
    """Choose the most robust segment length, breaking ties toward shorter windows."""
    best_min_group = max(r['min_group_true_best_rate'] for r in results)
    candidates = [r for r in results if r['min_group_true_best_rate'] == best_min_group]
    best_true_rate = max(r['true_best_rate'] for r in candidates)
    candidates = [r for r in candidates if r['true_best_rate'] == best_true_rate]
    best_margin = max(r['median_margin'] for r in candidates)
    candidates = [r for r in candidates if r['median_margin'] == best_margin]
    return min(candidates, key=lambda r: r['segment_len'])['segment_len']


def _print_segment_diag_summary(diag):
    """Print a compact summary of cached or freshly computed diagnostic results."""
    print('  segment_len diagnostic:')
    print(f'  {"samples":>8}  {"seconds":>8}  {"true_best":>10}  {"min_group":>10}  {"margin":>10}')
    for result in diag['results']:
        print(
            f"  {result['segment_len']:>8}  "
            f"{result['segment_len_s']:>8.3f}  "
            f"{result['true_best_rate']:>10.3f}  "
            f"{result['min_group_true_best_rate']:>10.3f}  "
            f"{result['median_margin']:>10.3e}"
        )
    print(f"  chosen_segment_len: {diag['chosen_segment_len']} samples")


def _get_or_run_segment_length_diagnostic(traj_specs, rmse_baseline, device, save_dir):
    """Load cached segment-length diagnostic or run it once and save the result."""
    param_sets = _build_segment_diag_param_sets()
    param_set_names = [item['name'] for item in param_sets]
    active_groups = _active_groups_from_specs(traj_specs)
    cache_path = _segment_diag_cache_path(save_dir, traj_specs)
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu')
        if _segment_diag_cache_matches(cached, traj_specs, param_set_names):
            print('  segment_len_diag: loaded from cache')
            _print_segment_diag_summary(cached)
            return cached

    print('  segment_len_diag: computing')
    trajs = _load_grouped_trajectories(
        traj_specs, TRAJ_DIR, _P_PHYSICS, _TS_PHYSICS, device, save_dir, load_tensors=True
    )
    diag_block = ParameterizedLFRBlock(RMSE_baseline=rmse_baseline).to(device)
    diag_wrapper = _SimWrapper(diag_block)
    results = []

    for candidate_s in SEGMENT_DIAG_CANDIDATES_S:
        segment_len = int(round(candidate_s / float(_TS_PHYSICS)))
        _attach_valid_start_idx(trajs, segment_len)
        group_counts = {group: SEGMENT_DIAG_SEGMENTS_PER_GROUP for group in active_groups}
        x0_seg, u_seg, q1_seg, sample_plan = _sample_balanced_segments(
            trajs, segment_len, group_counts, BASE_SEED + segment_len
        )

        losses = {
            param_set['name']: _eval_param_set_on_segments(
                diag_block, diag_wrapper, param_set['params'], x0_seg, u_seg, q1_seg
            )
            for param_set in param_sets
        }
        all_losses = torch.stack([losses[name] for name in param_set_names], dim=1)
        true_is_best = all_losses.argmin(dim=1) == 0
        detuned_losses = torch.stack([losses[name] for name in param_set_names[1:]], dim=1)
        margin = detuned_losses.min(dim=1).values - losses['true']

        group_true_best_rate = {}
        for group in active_groups:
            mask = torch.tensor([item['group'] == group for item in sample_plan], dtype=torch.bool)
            group_true_best_rate[group] = float(true_is_best[mask].double().mean().item())

        results.append({
            'segment_len': segment_len,
            'segment_len_s': candidate_s,
            'true_best_rate': float(true_is_best.double().mean().item()),
            'group_true_best_rate': group_true_best_rate,
            'min_group_true_best_rate': float(min(group_true_best_rate.values())),
            'median_margin': float(margin.median().item()),
            'mean_loss_by_set': {
                name: float(losses[name].mean().item()) for name in param_set_names
            },
            'sample_plan': sample_plan,
        })

    diag = {
        'version': SEGMENT_DIAG_VERSION,
        'traj_specs': tuple(traj_specs),
        'candidate_lengths_s': SEGMENT_DIAG_CANDIDATES_S,
        'candidate_lengths_samples': [result['segment_len'] for result in results],
        'n_per_group': SEGMENT_DIAG_SEGMENTS_PER_GROUP,
        'seed': BASE_SEED,
        'param_set_names': param_set_names,
        'results': results,
        'chosen_segment_len': _choose_segment_len_from_diag(results),
    }
    torch.save(diag, cache_path)
    _print_segment_diag_summary(diag)
    print(f'  segment_len_diag: saved to {cache_path}')
    return diag

def _print_param_detail(block, log_param_grads, lr):
    """Per-parameter learned value vs truth and log_param gradient norms."""
    with torch.no_grad():
        learned = block._recover_params()  # tuple of scalar tensors, _PARAM_NAMES order
    print(f'\n  [param detail  lr={lr:.2e}]')
    print(f'  {"Param":<6}  {"True":>10}  {"Learned":>10}  {"Delta%":>8}  {"|grad|":>10}')
    print(f'  {"-"*6}  {"-"*10}  {"-"*10}  {"-"*8}  {"-"*10}')
    for i, name in enumerate(_PARAM_NAMES):
        true_v = _TRUE_PARAMS[name]
        lrn_v  = float(learned[i])
        delta  = (lrn_v - true_v) / true_v * 100
        grad_v = float(log_param_grads[i].abs()) if log_param_grads is not None else float('nan')
        print(f'  {name:<6}  {true_v:>10.4f}  {lrn_v:>10.4f}  {delta:>+8.2f}%  {grad_v:>10.3e}')


def _save_profile(prof, save_dir):
    """Print profiler table to console and save to profile_out.txt."""
    table = prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=20)
    header = '=' * 60 + '\nProfiler - epoch 0 (top 20 ops by self-CPU time)\n' + '=' * 60
    path = os.path.join(save_dir, 'profile_out.txt')
    with open(path, 'w') as f:
        f.write(header + '\n' + table + '\n')
    print('\n' + header)
    print(table)
    print(f'  Saved text log to: {path}')
    
    # Export extensive Chrome trace
    trace_path = os.path.join(save_dir, 'profile_trace.json')
    prof.export_chrome_trace(trace_path)
    print(f'  Saved detailed Chrome Trace to: {trace_path} (open in chrome://tracing)')


class _SimWrapper(torch.nn.Module):
    """Thin wrapper so DataParallel can replicate block across GPUs."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x0_seg, u_seg):
        params = self.block._recover_params()
        kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh = params
        params_10 = torch.stack([kb1 + kb2, cg1, cg2, cy, cb1 + cb2, mh, m1, m2, mb, Jb + Jh])
        _, M1, M2, K, C = _build_matrices(params_10, self.block._Lb, self.block._d)
        alpha, beta, gamma, N0, N1, N2 = build_poly_constants(
            m1, m2, mb, mh, Jb, Jh, self.block._Lb, self.block._d
        )
        d0 = mh * (alpha * gamma - beta ** 2)
        G = build_G_matrix(N0, d0, M1, M2, K, C)
        return simulate(
            x0_seg,
            u_seg,
            G,
            K,
            C,
            mh,
            alpha,
            beta,
            gamma,
            N0,
            N1,
            N2,
            self.block._P,
            self.block._ts,
            bptt_mode='full',
            return_latents=False,
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
    epochs=EPOCHS,
    lr=LR,
    segment_len=SEGMENT_LEN,
    n_steps=N_STEPS,
    log_interval=LOG_INTERVAL,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    save_dir=SAVE_DIR,
    profile=PROFILE,
    time_epochs=TIME_EPOCHS,
    param_loss_weight=PARAM_LOSS_WEIGHT,
    split_reg_weight=SPLIT_REG_WEIGHT,
):
    """Run parameter recovery training. Returns trained ParameterizedLFRBlock."""
    os.makedirs(save_dir, exist_ok=True)
    traj_specs = _active_traj_specs()
    traj_tag = _traj_set_tag(traj_specs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'  Device: {torch.cuda.get_device_name(0)}  (CUDA {torch.version.cuda})')
    else:
        print('  Device: CPU')

    # ------------------------------------------------------------------
    # 1. RMSE_baseline (D-034)
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}\nStep 1: RMSE_baseline + normalisation sigma\n{"=" * 60}')
    rmse_baseline, _rmse_entries = _get_or_compute_rmse_baseline(traj_specs, device, save_dir)
    sigma_dict = _get_or_compute_sigma(save_dir)
    # Fuse mask and sigma into a single per-trajectory weight vector (mask/sigma).
    # Dormant channels: mask=0, so weight=0 — one multiply in the loss instead of divide+multiply.
    # n_active is a fixed integer per trajectory, derived purely from CHANNEL_MASKS.
    weight_device = {
        tid: (torch.tensor(CHANNEL_MASKS[tid], dtype=torch.float64) / sigma_dict[tid]).to(device)
        for tid in sigma_dict
    }
    n_active_per_traj = {tid: int(sum(CHANNEL_MASKS[tid])) for tid in CHANNEL_MASKS}
    print(f'  {"Traj":>4}  {"mask":>8}  {"sigma_X1[m]":>13}  {"sigma_X2[m]":>13}  {"sigma_Y[m]":>11}')
    for spec in traj_specs:
        tid = spec['id']
        s = sigma_dict[tid]
        m = CHANNEL_MASKS[tid]
        print(
            f'  {tid:>4}  {str(m):>8}  {s[0]:>13.4e}  {s[1]:>13.4e}  {s[2]:>11.4e}'
        )
    rmse_baseline_normalized = _aggregate_normalized_rmse_baseline(_rmse_entries, sigma_dict)
    print(f'  RMSE_baseline normalized: {rmse_baseline_normalized:.6e}')

    # ------------------------------------------------------------------
    # 2. Data
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}\nStep 2a: Prepare grouped trajectory caches\n{"=" * 60}')
    _load_grouped_trajectories(
        traj_specs, TRAJ_DIR, _P_PHYSICS, _TS_PHYSICS, device, save_dir, load_tensors=False
    )

    auto_segment = segment_len is None
    if auto_segment:
        print(f'\n{"=" * 60}\nStep 2b: Segment-length diagnostic\n{"=" * 60}')
        diag = _get_or_run_segment_length_diagnostic(traj_specs, rmse_baseline, device, save_dir)
        segment_len = int(diag['chosen_segment_len'])
        print(f'  Using segment_len={segment_len} from cached diagnostic result')

    if profile:
        n_steps = 500
    if n_steps is not None and segment_len > n_steps:
        segment_len = n_steps
    data_step = '2c' if auto_segment else '2b'
    print(f'\n{"=" * 60}\nStep {data_step}: Load active training trajectories\n{"=" * 60}')
    trajs = _load_grouped_trajectories(
        traj_specs, TRAJ_DIR, _P_PHYSICS, _TS_PHYSICS, device, save_dir,
        load_tensors=True, n_steps=n_steps,
    )
    _attach_valid_start_idx(trajs, segment_len)
    active_groups = _active_groups_from_trajs(trajs)
    train_group_counts = _balanced_group_counts(TRAIN_SEGMENTS_PER_EPOCH, active_groups)
    print(
        f'  Active set: {", ".join(spec["id"] for spec in traj_specs)}  '
        f'({len(trajs)} trajectories, {len(active_groups)} groups)'
    )
    print(
        f'  segment_len={segment_len}  '
        f'train batch={sum(train_group_counts.values())} segments/epoch'
    )

    # ------------------------------------------------------------------
    # 3. Block + optimizer
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}\nStep 3: Build model\n{"=" * 60}')
    # Pass the sigma-normalized RMSE (not the metre-space value) so that Lambda
    # is calibrated in the same unit system as mse_loss (D-034, D-044).
    block = ParameterizedLFRBlock(RMSE_baseline=rmse_baseline_normalized).to(device)
    optimizer = torch.optim.Adam(block.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=7,    # steps on eval_rmse (~every 10 training epochs); 7 eval-steps ≈ 70 training epochs
        factor=0.5,
        min_lr=1e-5,
    )
    n_gpus = min(4, torch.cuda.device_count()) if device.type == 'cuda' else 0
    # Reserve last GPU for async eval; give the rest to DataParallel.
    n_gpus_train = max(n_gpus - 1, 1)
    eval_device   = torch.device(f'cuda:{n_gpus - 1}') if n_gpus > 1 else device

    wrapper = (
        torch.nn.DataParallel(_SimWrapper(block), device_ids=list(range(n_gpus_train)))
        if n_gpus_train > 1 else _SimWrapper(block)
    )
    print(f'  Trainable params : {sum(p.numel() for p in block.parameters())}')
    print(f'  RMSE_baseline    : {rmse_baseline:.6e} m  (normalized: {rmse_baseline_normalized:.6e})')
    gpu_names = ", ".join(
        torch.cuda.get_device_name(i)
        for i in range(max(n_gpus, 1) if device.type == 'cuda' else 0)
    )
    print(f'  GPUs in use      : {n_gpus if n_gpus > 1 else 1}  ({gpu_names})\n')
    print(block.param_table())

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    print(
        f'\n{"=" * 60}\nStep 4: Train  '
        f'({epochs} epochs, lr={lr}, batch={sum(train_group_counts.values())}x{segment_len})\n{"=" * 60}'
    )
    if param_loss_weight > 0:
        print(
            f'  {"Epoch":>6}  {"train_rmse[m]":>14}  {"param_loss":>12}  '
            f'{"total":>12}  {"grad_norm":>12}  {"time [s]":>9}  {"lr":>10}  |  {"eval_ep":>7}  {"eval_rmse[m]":>13}'
        )
        print(f'  {"-" * 6}  {"-" * 14}  {"-" * 12}  {"-" * 12}  {"-" * 12}  {"-" * 9}  {"-" * 10}  |  {"-" * 7}  {"-" * 13}')
    else:
        print(
            f'  {"Epoch":>6}  {"train_rmse[m]":>14}  {"grad_norm":>12}  {"time [s]":>9}  {"lr":>10}  |  {"eval_ep":>7}  {"eval_rmse[m]":>13}'
        )
        print(f'  {"-" * 6}  {"-" * 14}  {"-" * 12}  {"-" * 9}  {"-" * 10}  |  {"-" * 7}  {"-" * 13}')

    t_start = time.time()
    history = []  # one entry per log_interval epoch
    latest_eval_epoch = '-'
    latest_eval_rmse  = '-'

    # Eval copies on eval_device. If eval_device == device (single GPU), .to() is a no-op.
    eval_trajs = []
    for t in trajs:
        eval_t = {}
        for k, v in t.items():
            if isinstance(v, torch.Tensor):
                eval_t[k] = v.to(eval_device)
            else:
                eval_t[k] = v
        eval_trajs.append(eval_t)

    # Best-epoch tracking (updated by full-traj eval results)
    best_full_traj_rmse = float('inf')
    best_epoch          = -1
    best_log_params     = None

    # Async eval worker (only when a dedicated GPU is available)
    use_async_eval = (n_gpus > 1)
    snap_queue   = queue.Queue(maxsize=2)   # main -> worker: (epoch, log_params_cpu)
    result_queue = queue.Queue()            # worker -> main: (epoch, rmse, entries, log_params_cpu)

    if use_async_eval:
        _eval_block = ParameterizedLFRBlock(RMSE_baseline=rmse_baseline_normalized).to(eval_device)

        def _eval_worker():
            while True:
                item = snap_queue.get()
                if item is None:  # poison pill
                    snap_queue.task_done()  # unblock snap_queue.join() in post-loop cleanup
                    break
                snap_epoch, lp_cpu = item
                with torch.no_grad():
                    _eval_block.log_params.copy_(lp_cpu.to(eval_device))
                rmse, entries = _full_traj_eval(_eval_block, eval_trajs)
                result_queue.put((snap_epoch, rmse, entries, lp_cpu))
                snap_queue.task_done()

        threading.Thread(target=_eval_worker, daemon=True).start()

    prof = None
    if profile:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=True,
        )
        prof.start()

    for epoch in range(epochs):
        with torch.profiler.record_function(f"Epoch {epoch}") if profile else contextlib.nullcontext():
            t0 = _sync_time(device)
            optimizer.zero_grad()
    
            x0_seg, u_seg, q1_seg, sample_plan = _sample_balanced_segments(
                trajs, segment_len, train_group_counts, BASE_SEED + 10_000 + epoch
            )
            # Precompute batch weight/n_active from sample_plan — fixed constants, no grad.
            batch_weights = torch.stack(
                [weight_device[p['traj_id']] for p in sample_plan]
            )  # (B, 3): mask/sigma fused; dormant channels are 0
            batch_n_active = torch.tensor(
                [n_active_per_traj[p['traj_id']] for p in sample_plan],
                dtype=torch.float64, device=device,
            )  # (B,)
    
            Y_pred = wrapper(x0_seg, u_seg)
            # Per-segment masked + normalized loss (D-044, docs/loss-function-design.md).
            # Single multiply by weight (= mask/sigma) masks dormant channels and
            # normalizes active ones; per-segment average over (n_active * T).
            err = (Y_pred - q1_seg) * batch_weights.unsqueeze(1)               # (B, T, 3)
            seg_losses = err.pow(2).sum(dim=(1, 2)) / (batch_n_active * segment_len)  # (B,)
            mse_loss = seg_losses.mean()
            theta_loss = block.param_loss() if param_loss_weight > 0 else None
            split_reg = block.split_loss() * split_reg_weight if split_reg_weight > 0 else 0.0
            loss = mse_loss + split_reg + (param_loss_weight * theta_loss if theta_loss is not None else 0)
            t_fwd = _sync_time(device)
            loss.backward()
            t_bwd = _sync_time(device)

        if block.log_params.grad is not None:
            _pg = block.log_params.grad.detach().clone()
            grad_norm = _pg.norm().item()
        else:
            _pg = None
            grad_norm = float('nan')
        optimizer.step()

        if checkpoint_interval > 0 and epoch > 0 and epoch % checkpoint_interval == 0:
            torch.save(
                {'log_params': block.log_params.detach(), 'epoch': epoch, 'history': history},
                os.path.join(save_dir, f'checkpoint_e{epoch}.pt'),
            )

        # ── Async path: drain completed results (non-blocking) ──────────────────────
        if use_async_eval:
            while not result_queue.empty():
                snap_epoch, full_rmse, _, lp_cpu = result_queue.get_nowait()
                latest_eval_epoch = snap_epoch
                latest_eval_rmse = f"{full_rmse:.4e}"
                scheduler.step(full_rmse)
                # Back-fill full_traj_rmse into the history entry for snap_epoch
                for h in history:
                    if h['epoch'] == snap_epoch:
                        h['full_traj_rmse_m'] = full_rmse
                        h['log_params_snapshot'] = lp_cpu
                        break
                if full_rmse < best_full_traj_rmse:
                    best_full_traj_rmse, best_epoch, best_log_params = full_rmse, snap_epoch, lp_cpu

            # Push snapshot every FULL_EVAL_INTERVAL epochs
            if epoch % FULL_EVAL_INTERVAL == 0 or epoch == epochs - 1:
                try:
                    if epoch == epochs - 1:
                        snap_queue.put((epoch, block.log_params.detach().cpu().clone()))
                    else:
                        snap_queue.put_nowait((epoch, block.log_params.detach().cpu().clone()))
                except queue.Full:
                    pass  # worker not yet done with previous snapshot; skip (rare with 80% slack)

        # ── Sync path (single GPU / CPU): block and run directly ────────────────────
        elif epoch % FULL_EVAL_INTERVAL == 0 or epoch == epochs - 1:
            full_rmse, _ = _full_traj_eval(block, eval_trajs)
            latest_eval_epoch = epoch
            latest_eval_rmse = f"{full_rmse:.4e}"
            scheduler.step(full_rmse)
            if full_rmse < best_full_traj_rmse:
                best_full_traj_rmse = full_rmse
                best_epoch          = epoch
                best_log_params     = block.log_params.detach().cpu().clone()

        if epoch % log_interval == 0 or epoch == epochs - 1:
            train_err = Y_pred.detach() - q1_seg
            train_rmse_m = train_err.pow(2).mean().sqrt().item()
            train_rmse_ch = train_err.pow(2).mean(dim=(0, 1)).sqrt().cpu()  # (3,)
            current_lr = optimizer.param_groups[0]['lr']
            hist_entry = {
                'epoch': epoch,
                'train_rmse_m': train_rmse_m,
                'train_rmse_ch': train_rmse_ch,
                'mse_loss_norm': mse_loss.item(),
                'grad_norm': grad_norm,
                'lr': current_lr,
            }
            # For the sync path we can write the current epoch's full_rmse immediately
            if epoch % FULL_EVAL_INTERVAL == 0 or epoch == epochs - 1:
                if not use_async_eval:
                    hist_entry['full_traj_rmse_m'] = full_rmse
                    hist_entry['log_params_snapshot'] = block.log_params.detach().cpu().clone()

            if split_reg_weight > 0:
                hist_entry['split_reg'] = split_reg.item() if hasattr(split_reg, 'item') else float(split_reg)
            if param_loss_weight > 0:
                hist_entry['param_loss'] = theta_loss.item()
                hist_entry['total_loss'] = loss.item()
                print(
                    f'  {epoch:>6}  {train_rmse_m:>14.4e}  {theta_loss.item():>12.4e}  '
                    f'{loss.item():>12.4e}  {grad_norm:>12.3e}  {time.time() - t0:>9.3f}  {current_lr:>10.3e}  |  '
                    f'{latest_eval_epoch:>7}  {latest_eval_rmse:>13}',
                    flush=True,
                )
            else:
                print(
                    f'  {epoch:>6}  {train_rmse_m:>14.4e}  {grad_norm:>12.3e}  {time.time() - t0:>9.3f}  {current_lr:>10.3e}  |  '
                    f'{latest_eval_epoch:>7}  {latest_eval_rmse:>13}',
                    flush=True,
                )
            history.append(hist_entry)

        if time_epochs:
            print(
                f'    fwd={t_fwd - t0:.2f}s  bwd={t_bwd - t_fwd:.2f}s  total={t_bwd - t0:.2f}s',
                flush=True,
            )

    if prof is not None:
        prof.stop()
        _save_profile(prof, save_dir)

    if use_async_eval:
        snap_queue.put(None)          # poison pill -> worker exits cleanly
        snap_queue.join()             # wait for worker to finish current item
        # Drain any results that arrived during the final epochs
        while not result_queue.empty():
            snap_epoch, full_rmse, _, lp_cpu = result_queue.get_nowait()
            for h in history:
                if h['epoch'] == snap_epoch:
                    h['full_traj_rmse_m'] = full_rmse
                    h['log_params_snapshot'] = lp_cpu
                    break
            if full_rmse < best_full_traj_rmse:
                best_full_traj_rmse, best_epoch, best_log_params = full_rmse, snap_epoch, lp_cpu

    if epochs > 1:
        total = time.time() - t_start
        print(f'\n  Done: {total:.1f} s  ({total / epochs:.2f} s/epoch)')

    # ------------------------------------------------------------------
    # 5. Evaluate - fresh post-training pre-pass (pre may be stale)
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}\nStep 5: Prediction error  (best epoch = {best_epoch})\n{"=" * 60}')

    # Restore best-epoch parameters before the final eval pass
    if best_log_params is not None:
        with torch.no_grad():
            block.log_params.copy_(best_log_params.to(device))
        print(f'  Loaded best_log_params from epoch {best_epoch} '
              f'(full-traj RMSE = {best_full_traj_rmse:.4e} m)\n')

    eval_entries = []
    print(f'  {"Traj":<6}  {"Group":<12}  {"RMSE [m]":>12}  {"X1 [m]":>10}  {"X2 [m]":>10}  {"Y [m]":>10}')
    print(f'  {"-" * 6}  {"-" * 12}  {"-" * 12}  {"-" * 10}  {"-" * 10}  {"-" * 10}')
    for traj in trajs:
        pre = _run_no_grad(block, traj['state_traj'][:1], traj['u'])
        y_pred = pre.Y[0]
        diff = y_pred - traj['q1']
        mse_eval = diff.pow(2).mean().item()
        rmse_eval = mse_eval ** 0.5
        rmse_ch = diff.pow(2).mean(dim=0).sqrt().cpu()  # (3,) per channel
        eval_entries.append({
            'id': traj['id'],
            'group': traj['group'],
            'mse_total': mse_eval,
            'rmse_total': rmse_eval,
            'rmse_ch': rmse_ch,
        })
        print(
            f'  {traj["id"]:<6}  {traj["group"]:<12}  {rmse_eval:>12.4e}'
            f'  {rmse_ch[0]:>10.4e}  {rmse_ch[1]:>10.4e}  {rmse_ch[2]:>10.4e}'
        )
    overall_rmse = _aggregate_grouped_rmse(eval_entries)
    print(f'\n  Overall RMSE: {overall_rmse:.6e} m')

    # ------------------------------------------------------------------
    # 6. Parameter recovery table - primary go/no-go criterion
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}\nStep 6: Parameter recovery\n{"=" * 60}')
    print(block.param_table())

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    params_true = torch.tensor([_TRUE_PARAMS[n] for n in _PARAM_NAMES], dtype=torch.float64)
    params_learned = block.params_init * block.log_params.detach().exp()

    # Group-balanced per-channel RMSE (mirrors _aggregate_grouped_rmse logic)
    _group_mse_ch = {}
    for _e in eval_entries:
        _group_mse_ch.setdefault(_e['group'], []).append(_e['rmse_ch'].pow(2))
    eval_rmse_ch = (
        sum(torch.stack(v).mean(dim=0) for v in _group_mse_ch.values()) / len(_group_mse_ch)
    ).sqrt()  # (3,) group-balanced per-channel RMSE

    save_path = os.path.join(save_dir, f'lfr_param_recovery_{traj_tag}_e{epochs}_plw{param_loss_weight:.1f}.pt')
    torch.save(
        {
            # Parameters
            'param_names': list(_PARAM_NAMES),
            'params_true': params_true,
            'params_init': block.params_init,
            'params_learned': params_learned,
            'log_params': block.log_params.detach(),
            # Best-epoch tracking
            'best_epoch': best_epoch,
            'best_full_traj_rmse': best_full_traj_rmse,
            'best_log_params': best_log_params,
            # Normalisation
            'RMSE_baseline': rmse_baseline,
            'RMSE_baseline_normalized': rmse_baseline_normalized,
            'rmse_baseline_entries': _rmse_entries,
            'sigma': {tid: s.cpu() for tid, s in sigma_dict.items()},
            # Run config
            'active_traj_ids': tuple(spec['id'] for spec in traj_specs),
            'epochs': epochs,
            'lr': lr,
            'segment_len': segment_len,
            'param_loss_weight': param_loss_weight,
            'split_reg_weight': split_reg_weight,
            'train_segments_per_epoch': TRAIN_SEGMENTS_PER_EPOCH,
            'base_seed': BASE_SEED,
            # Results
            'eval_rmse': overall_rmse,
            'eval_rmse_ch': eval_rmse_ch,
            'eval_entries': eval_entries,
            'history': history,
        },
        save_path,
    )
    print(f'\n  Saved to: {save_path}')

    return block


if __name__ == '__main__':
    train(profile=PROFILE, time_epochs=TIME_EPOCHS)
