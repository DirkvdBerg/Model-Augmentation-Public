"""
train_param_recovery.py
-----------------------
Step 3b: recover true physical parameters from MATLAB data using batched multiple
shooting.

Approach:
    x0 is known exactly ([0, 0, 0.3, 0, 0, 0] logical), so no encoder is needed.
    Full state trajectories (positions + finite-difference velocities) are cached
    once per source trajectory in logical coordinates.
    At startup, grouped parameter-recovery trajectories are load-or-build cached so
    later multi-trajectory work does not recompute them.
    Each epoch: n_seg segments sampled via stratified random indexing (one per
    stratum), guaranteeing full trajectory coverage every epoch. Segment start states
    come from cached state_traj; u_seg and q1_seg use vectorized advanced indexing.
    Loss: MSE(Y_pred, q1_train) + block.param_loss()
    Optimizer: Adam on block.log_params only.

Current training data: Matlab-output/lpv_sim_varying_y.mat
    True-parameter trajectory (ground truth, D-033).
    u_q1 (N, 3) stage forces [N], q1 (N, 3) stage positions [m], fs = 20 kHz.

Grouped cache preparation data: Matlab-output/parameter-recovery/*.mat

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
MAT_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'Matlab-output', 'lpv_sim_varying_y.mat'
)
TRAJ_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'Matlab-output', 'parameter-recovery'
)
SAVE_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'models', 'gantry', 'param_recovery'
)

TRAJ_SPECS = (
    {'id': 'T1', 'group': 'y_only', 'file': 'T1_Y_sweep_conservative.mat'},
    {'id': 'T6', 'group': 'y_only', 'file': 'T6_Y_sweep_aggressive.mat'},
    {'id': 'T2', 'group': 'x_sym_mh', 'file': 'T2_X_sym_Y030.mat'},
    {'id': 'T3', 'group': 'x_sym_mh', 'file': 'T3_X_sym_Y000.mat'},
    {'id': 'T4', 'group': 'rot_coupled', 'file': 'T4_X_antisym_Y020.mat'},
    {'id': 'T5', 'group': 'rot_coupled', 'file': 'T5_X_sym_Y_sweep.mat'},
)
TRAJ_GROUPS = ('y_only', 'x_sym_mh', 'rot_coupled')

N_STEPS = None  # cap on steps (None = use all); overridden to 500 when PROFILE=True
EPOCHS = 3
LR = 1e-3
SEGMENT_LEN = 4000
PARAM_LOSS_WEIGHT = 0.0
LOG_INTERVAL = 25
CHECKPOINT_INTERVAL = 100
PROFILE = False
TIME_EPOCHS = False
USE_MULTI_TRAJ_BASELINE = False
AUTO_SEGMENT_LEN = False
SEGMENT_DIAG_CANDIDATES_S = (0.1, 0.2, 0.4, 0.6)
SEGMENT_DIAG_N_PER_GROUP = 8
SEGMENT_DIAG_SEED = 1234
SEGMENT_DIAG_VERSION = 1
RMSE_BASELINE_CACHE_VERSION = 1


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


def _load_grouped_trajectories(traj_specs, traj_dir, P, ts, device, save_dir, load_tensors):
    """Load grouped trajectories and ensure per-trajectory state caches exist."""
    trajs = []
    grouped = {group: [] for group in TRAJ_GROUPS}
    for spec in traj_specs:
        mat_path = os.path.join(traj_dir, spec['file'])
        u_i, q1_i, fs_i = _load_trajectory(mat_path)
        state_traj_i = _load_or_build_state_traj(
            spec['id'], q1_i, P, ts, device, save_dir, load=load_tensors
        )
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


def _baseline_specs_for_run(mat_path, use_multi_traj_baseline):
    """Select which trajectories contribute to RMSE_baseline for this run."""
    if use_multi_traj_baseline:
        return tuple(
            {
                'id': spec['id'],
                'group': spec['group'],
                'file': spec['file'],
                'path': os.path.join(TRAJ_DIR, spec['file']),
            }
            for spec in TRAJ_SPECS
        )

    file_name = os.path.basename(mat_path)
    return ({
        'id': os.path.splitext(file_name)[0],
        'group': 'single',
        'file': file_name,
        'path': mat_path,
    },)


def _load_rmse_baseline_cache(save_dir):
    """Load the RMSE_baseline cache or create an empty one."""
    cache_path = _rmse_baseline_cache_path(save_dir)
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu')
        if cached.get('version') == RMSE_BASELINE_CACHE_VERSION:
            return cached
    return {'version': RMSE_BASELINE_CACHE_VERSION, 'per_traj': {}}


def _aggregate_rmse_baseline(selected_entries):
    """Aggregate cached per-trajectory baseline results into one scalar."""
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
        print(
            f"  {entry['id']}: RMSE = {entry['rmse_total']:.6e} m  "
            f"({entry['rmse_total'] * 1e3:.4f} mm)"
        )
    else:
        print(f'  {"Traj":<6}  {"Group":<12}  {"RMSE [mm]":>12}')
        print(f'  {"-" * 6}  {"-" * 12}  {"-" * 12}')
        for entry in selected_entries:
            print(
                f"  {entry['id']:<6}  {entry['group']:<12}  "
                f"{entry['rmse_total'] * 1e3:>12.4f}"
            )
    print(f'  Overall RMSE_baseline = {overall_rmse:.6e} m  ({overall_rmse * 1e3:.4f} mm)')


def _get_or_compute_rmse_baseline(mat_path, use_multi_traj_baseline, device, save_dir):
    """Load cached per-trajectory RMSE_baseline metrics and derive the run scalar."""
    specs = _baseline_specs_for_run(mat_path, use_multi_traj_baseline)
    cache = _load_rmse_baseline_cache(save_dir)
    updated = False
    selected_entries = []

    for spec in specs:
        entry = cache['per_traj'].get(spec['id'])
        if entry is None or entry.get('file') != spec['file'] or entry.get('group') != spec['group']:
            _, q1_i, _ = _load_trajectory(spec['path'])
            state_traj_i = _load_or_build_state_traj(
                spec['id'], q1_i, _P_PHYSICS, _TS_PHYSICS, device, save_dir, load=True
            )
            metrics = compute_rmse_baseline_metrics(
                mat_path=spec['path'],
                x0_logical=state_traj_i[:1],
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

    overall_rmse = _aggregate_rmse_baseline(selected_entries)
    _print_rmse_baseline_summary(selected_entries, overall_rmse)
    return overall_rmse


def _attach_valid_start_idx(trajs, segment_len):
    """Attach valid segment starts 0 .. N - segment_len for each trajectory."""
    for traj in trajs:
        n_valid = max(traj['N'] - segment_len + 1, 0)
        traj['valid_start_idx'] = torch.arange(n_valid, dtype=torch.int64)


def _sample_balanced_segments(trajs, segment_len, n_per_group, seed):
    """Sample the same number of segments from each trajectory group."""
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)

    x0_seg_list = []
    u_seg_list = []
    q1_seg_list = []
    sample_plan = []

    for group in TRAJ_GROUPS:
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


def _segment_diag_cache_path(save_dir):
    """Cache path for the segment-length diagnostic result."""
    return os.path.join(save_dir, f'segment_len_diag_v{SEGMENT_DIAG_VERSION}.pt')


def _segment_diag_cache_matches(cached, param_set_names):
    """Return True when the cached diagnostic matches the current config."""
    return (
        cached.get('version') == SEGMENT_DIAG_VERSION
        and tuple(cached.get('traj_specs', ())) == TRAJ_SPECS
        and tuple(cached.get('candidate_lengths_s', ())) == SEGMENT_DIAG_CANDIDATES_S
        and cached.get('n_per_group') == SEGMENT_DIAG_N_PER_GROUP
        and cached.get('seed') == SEGMENT_DIAG_SEED
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


def _get_or_run_segment_length_diagnostic(rmse_baseline, device, save_dir):
    """Load cached segment-length diagnostic or run it once and save the result."""
    param_sets = _build_segment_diag_param_sets()
    param_set_names = [item['name'] for item in param_sets]
    cache_path = _segment_diag_cache_path(save_dir)
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu')
        if _segment_diag_cache_matches(cached, param_set_names):
            print('  segment_len_diag: loaded from cache')
            _print_segment_diag_summary(cached)
            return cached

    print('  segment_len_diag: computing')
    trajs = _load_grouped_trajectories(
        TRAJ_SPECS, TRAJ_DIR, _P_PHYSICS, _TS_PHYSICS, device, save_dir, load_tensors=True
    )
    diag_block = ParameterizedLFRBlock(RMSE_baseline=rmse_baseline).to(device)
    diag_wrapper = _SimWrapper(diag_block)
    results = []

    for candidate_s in SEGMENT_DIAG_CANDIDATES_S:
        segment_len = int(round(candidate_s / float(_TS_PHYSICS)))
        _attach_valid_start_idx(trajs, segment_len)
        x0_seg, u_seg, q1_seg, sample_plan = _sample_balanced_segments(
            trajs, segment_len, SEGMENT_DIAG_N_PER_GROUP, SEGMENT_DIAG_SEED + segment_len
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
        for group in TRAJ_GROUPS:
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
        'traj_specs': TRAJ_SPECS,
        'candidate_lengths_s': SEGMENT_DIAG_CANDIDATES_S,
        'candidate_lengths_samples': [result['segment_len'] for result in results],
        'n_per_group': SEGMENT_DIAG_N_PER_GROUP,
        'seed': SEGMENT_DIAG_SEED,
        'param_set_names': param_set_names,
        'results': results,
        'chosen_segment_len': _choose_segment_len_from_diag(results),
    }
    torch.save(diag, cache_path)
    _print_segment_diag_summary(diag)
    print(f'  segment_len_diag: saved to {cache_path}')
    return diag


_EPOCH_CACHE_SIZE = 10_000


def _get_epoch_indices(N_steps, n_seg, segment_len, epochs, device, save_dir):
    """
    Stratified random segment start indices for all training epochs.

    Trajectory [0, N_steps - segment_len] is split into n_seg equal strata;
    one random start is sampled per stratum per epoch.
    """
    tag = f'n{N_steps}_sl{segment_len}_nb{n_seg}'
    cache_path = os.path.join(save_dir, f'epoch_idx_{tag}.pt')
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu')
        if cached.shape[0] >= epochs:
            print(f'  epoch_idx: loaded from cache  ({tag})')
            return cached.to(device)

    n_gen = max(_EPOCH_CACHE_SIZE, epochs)
    stratum = (N_steps - segment_len) // n_seg
    base = torch.arange(n_seg).unsqueeze(0) * stratum
    offsets = torch.randint(0, stratum, (n_gen, n_seg))
    idx = (base + offsets).to(torch.int64)
    torch.save(idx.cpu(), cache_path)
    print(f'  epoch_idx: computed and cached  ({tag}, {n_gen} epochs)')
    return idx.to(device)


def _save_profile(prof, save_dir):
    """Print profiler table to console and save to profile_out.txt."""
    table = prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=20)
    header = '=' * 60 + '\nProfiler - epoch 0 (top 20 ops by self-CPU time)\n' + '=' * 60
    path = os.path.join(save_dir, 'profile_out.txt')
    with open(path, 'w') as f:
        f.write(header + '\n' + table + '\n')
    print('\n' + header)
    print(table)
    print(f'  Saved to: {path}')


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
    mat_path=MAT_PATH,
    save_dir=SAVE_DIR,
    profile=PROFILE,
    time_epochs=TIME_EPOCHS,
    param_loss_weight=PARAM_LOSS_WEIGHT,
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
    print(f'\n{"=" * 60}\nStep 1: RMSE_baseline\n{"=" * 60}')
    rmse_baseline = _get_or_compute_rmse_baseline(
        mat_path, USE_MULTI_TRAJ_BASELINE, device, save_dir
    )

    # ------------------------------------------------------------------
    # 2. Data
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}\nStep 2a: Prepare grouped trajectory caches\n{"=" * 60}')
    _load_grouped_trajectories(
        TRAJ_SPECS, TRAJ_DIR, _P_PHYSICS, _TS_PHYSICS, device, save_dir, load_tensors=False
    )

    if AUTO_SEGMENT_LEN:
        print(f'\n{"=" * 60}\nStep 2b: Segment-length diagnostic\n{"=" * 60}')
        diag = _get_or_run_segment_length_diagnostic(rmse_baseline, device, save_dir)
        segment_len = int(diag['chosen_segment_len'])
        print(f'  Using segment_len={segment_len} from cached diagnostic result')

    data_step = '2c' if AUTO_SEGMENT_LEN else '2b'
    print(f'\n{"=" * 60}\nStep {data_step}: Load and segment training data\n{"=" * 60}')
    u_train, q1_train, _ = _load_trajectory(mat_path)
    if profile:
        n_steps = 500
    if n_steps is not None:
        u_train = u_train[:, :n_steps, :]
        q1_train = q1_train[:n_steps]
    u_train = u_train.to(device)
    q1_train = q1_train.to(device)

    N_steps = u_train.shape[1]
    n_seg = N_steps // segment_len
    print(
        f'  {N_steps} steps -> {n_seg} x {segment_len} per epoch  '
        f'(stratified random, {N_steps - segment_len + 1} valid start positions)'
    )

    # ------------------------------------------------------------------
    # 3. Block + optimizer
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}\nStep 3: Build model\n{"=" * 60}')
    block = ParameterizedLFRBlock(RMSE_baseline=rmse_baseline).to(device)
    optimizer = torch.optim.Adam(block.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5,
        min_lr=1e-5,
    )
    n_gpus = min(4, torch.cuda.device_count()) if device.type == 'cuda' else 0
    wrapper = (
        torch.nn.DataParallel(_SimWrapper(block), device_ids=list(range(n_gpus)))
        if n_gpus > 1 else _SimWrapper(block)
    )
    print(f'  Trainable params : {sum(p.numel() for p in block.parameters())}')
    print(f'  RMSE_baseline    : {rmse_baseline:.6e} m')
    gpu_names = ", ".join(
        torch.cuda.get_device_name(i)
        for i in range(max(n_gpus, 1) if device.type == 'cuda' else 0)
    )
    print(f'  GPUs in use      : {n_gpus if n_gpus > 1 else 1}  ({gpu_names})\n')
    print(block.param_table())

    # ------------------------------------------------------------------
    # 3b. Precomputed data structures - parameter-free
    # ------------------------------------------------------------------
    traj_id = os.path.splitext(os.path.basename(mat_path))[0]
    state_traj = _load_or_build_state_traj(traj_id, q1_train, block._P, block._ts, device, save_dir)
    x0 = state_traj[:1]
    all_idx = _get_epoch_indices(N_steps, n_seg, segment_len, epochs, device, save_dir)[:epochs]
    _arange_T = torch.arange(segment_len, device=device)
    val_idx = torch.arange(n_seg, device=device) * segment_len
    val_step = val_idx.unsqueeze(1) + _arange_T
    val_x0 = state_traj[val_idx]
    val_u = u_train[0][val_step]
    val_q1 = q1_train[val_step]

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}\nStep 4: Train  ({epochs} epochs, lr={lr}, batch={n_seg}x{segment_len})\n{"=" * 60}')
    if param_loss_weight > 0:
        print(
            f'  {"Epoch":>6}  {"train_mse":>12}  {"param_loss":>12}  '
            f'{"total":>12}  {"val_mse":>12}  {"grad_norm":>12}  {"time [s]":>9}'
        )
        print(f'  {"-" * 6}  {"-" * 12}  {"-" * 12}  {"-" * 12}  {"-" * 12}  {"-" * 12}  {"-" * 9}')
    else:
        print(f'  {"Epoch":>6}  {"train_mse":>12}  {"val_mse":>12}  {"grad_norm":>12}  {"time [s]":>9}')
        print(f'  {"-" * 6}  {"-" * 12}  {"-" * 12}  {"-" * 12}  {"-" * 9}')

    t_start = time.time()

    for epoch in range(epochs):
        t0 = _sync_time(device)
        optimizer.zero_grad()

        ctx = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=False,
                with_stack=False,
            ) if (profile and epoch == 0) else contextlib.nullcontext()
        )
        step_idx = all_idx[epoch].unsqueeze(1) + _arange_T
        x0_seg = state_traj[all_idx[epoch]]
        u_seg = u_train[0][step_idx]
        q1_seg = q1_train[step_idx]

        with ctx as prof:
            Y_pred = wrapper(x0_seg, u_seg)
            mse_loss = F.mse_loss(Y_pred, q1_seg)
            theta_loss = block.param_loss() if param_loss_weight > 0 else None
            loss = mse_loss + (param_loss_weight * theta_loss if theta_loss is not None else 0)
            t_fwd = _sync_time(device)
            loss.backward()
            t_bwd = _sync_time(device)

        if prof is not None:
            _save_profile(prof, save_dir)

        grad_norm = (
            block.log_params.grad.norm().item()
            if block.log_params.grad is not None else float('nan')
        )
        optimizer.step()

        if checkpoint_interval > 0 and epoch > 0 and epoch % checkpoint_interval == 0:
            torch.save(
                {'log_params': block.log_params.detach(), 'epoch': epoch},
                os.path.join(save_dir, f'checkpoint_e{epoch}.pt'),
            )

        if epoch % log_interval == 0 or epoch == epochs - 1:
            with torch.no_grad():
                val_mse = F.mse_loss(wrapper(val_x0, val_u), val_q1).item()
            scheduler.step(val_mse)
            if param_loss_weight > 0:
                print(
                    f'  {epoch:>6}  {mse_loss.item():>12.4e}  {theta_loss.item():>12.4e}  '
                    f'{loss.item():>12.4e}  {val_mse:>12.4e}  {grad_norm:>12.3e}  {time.time() - t0:>9.3f}',
                    flush=True,
                )
            else:
                print(
                    f'  {epoch:>6}  {mse_loss.item():>12.4e}  {val_mse:>12.4e}  '
                    f'{grad_norm:>12.3e}  {time.time() - t0:>9.3f}',
                    flush=True,
                )
        if time_epochs:
            print(
                f'    fwd={t_fwd - t0:.2f}s  bwd={t_bwd - t_fwd:.2f}s  total={t_bwd - t0:.2f}s',
                flush=True,
            )

    if epochs > 1:
        total = time.time() - t_start
        print(f'\n  Done: {total:.1f} s  ({total / epochs:.2f} s/epoch)')

    # ------------------------------------------------------------------
    # 5. Evaluate - fresh post-training pre-pass (pre may be stale)
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}\nStep 5: Prediction error\n{"=" * 60}')
    pre = _run_no_grad(block, x0, u_train)
    y_pred = pre.Y[0]
    mse_eval = F.mse_loss(y_pred, q1_train).item()
    err = (y_pred - q1_train).pow(2).mean(0).sqrt()
    print(f'  {"Channel":<6}  {"RMSE [mm]":>12}')
    print(f'  {"-" * 6}  {"-" * 12}')
    for name, e in zip(['X1', 'X2', 'Y'], err):
        print(f'  {name:<6}  {e.item() * 1e3:>12.4f}')
    print(f'\n  Overall MSE: {mse_eval:.4e} m^2')

    # ------------------------------------------------------------------
    # 6. Parameter recovery table - primary go/no-go criterion
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}\nStep 6: Parameter recovery\n{"=" * 60}')
    print(block.param_table())

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    params_true = torch.tensor([_TRUE_PARAMS[n] for n in _PARAM_NAMES], dtype=torch.float64)
    save_path = os.path.join(save_dir, f'lfr_param_recovery_e{epochs}_plw{param_loss_weight:.1f}.pt')
    torch.save(
        {
            'log_params': block.log_params.detach(),
            'params_init': block.params_init,
            'params_true': params_true,
            'RMSE_baseline': rmse_baseline,
            'epochs': epochs,
            'lr': lr,
            'segment_len': segment_len,
            'param_loss_weight': param_loss_weight,
            'train_mse': mse_eval,
        },
        save_path,
    )
    print(f'\n  Saved to: {save_path}')

    return block


if __name__ == '__main__':
    train(profile=PROFILE, time_epochs=TIME_EPOCHS)
