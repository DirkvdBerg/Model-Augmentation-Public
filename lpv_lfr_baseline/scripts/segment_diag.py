"""
segment_diag.py
---------------
Segment-length diagnostic for windowed BPTT parameter recovery.

Run once to choose the optimal window length W for the outer BPTT loop in
train_param_recovery.py. Tests candidate lengths [0.1, 0.2, 0.4, 0.6] seconds
and scores each by whether the true parameters produce lower loss than detuned
alternatives on held-out segments. Result is cached to disk.

Interface
---------
run_segment_diag(traj_specs, traj_dir, save_dir) -> int
    Load cached result if available; otherwise run and cache. Returns the
    chosen segment length in samples.

Called by precompute.py, which stores the result as segment_len in its cache.
"""

import os

import torch

from lpv_lfr_baseline.blocks.lfr_param_block import (
    ParameterizedLFRBlock,
    _PARAM_NAMES,
    _TRUE_PARAMS,
    _build_matrices,
)
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.lfr_simulate import simulate
from lpv_lfr_baseline.core.physics import P as _P, ts as _ts, build_poly_constants
from lpv_lfr_baseline.scripts.precompute import _build_state_traj_logical, _load_trajectory

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

SEGMENT_DIAG_CANDIDATES_S    = (0.1, 0.2, 0.4, 0.6)
SEGMENT_DIAG_N_SEGMENTS = 24
SEGMENT_DIAG_VERSION         = 1
_BASE_SEED                   = 1234


# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------

def _traj_set_tag(traj_specs):
    """Stable tag for cache filenames derived from the active trajectory ids."""
    return '-'.join(spec['id'] for spec in traj_specs)


def _load_trajs(traj_specs, traj_dir, dtype):
    """Load trajectories and build state trajectories. Returns list of dicts."""
    P      = _P.to(dtype)
    ts_val = float(_ts)
    trajs  = []
    for spec in traj_specs:
        mat_path = os.path.join(traj_dir, spec['file'])
        u, q1, fs = _load_trajectory(mat_path, dtype)
        state_traj = _build_state_traj_logical(q1, P, ts_val, dtype)
        trajs.append({
            'id':         spec['id'],
            'N':          int(q1.shape[0]),
            'fs':         fs,
            'u':          u,            # (1, T, 3)
            'q1':         q1,           # (T, 3)
            'state_traj': state_traj,   # (T, 6)
        })
    return trajs


def _attach_valid_start_idx(trajs, segment_len):
    """Attach valid segment start indices 0 .. N - segment_len for each trajectory."""
    for traj in trajs:
        n_valid = max(traj['N'] - segment_len + 1, 0)
        traj['valid_start_idx'] = torch.arange(n_valid, dtype=torch.int64)


def _sample_balanced_segments(trajs, segment_len, n_segments, seed):
    """
    Sample n_segments uniformly from all trajectories with valid starts.

    Pre-allocates (n_segments, ...) output tensors and fills by index.
    """
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)

    valid_trajs = [t for t in trajs if t['valid_start_idx'].numel() > 0]
    if not valid_trajs:
        raise ValueError(f'No valid trajectories at segment_len={segment_len}')

    ref    = valid_trajs[0]
    dtype  = ref['state_traj'].dtype
    device = ref['state_traj'].device

    x0_seg    = torch.empty(n_segments, 6,              dtype=dtype, device=device)
    u_seg     = torch.empty(n_segments, segment_len, 3, dtype=dtype, device=device)
    q1_seg    = torch.empty(n_segments, segment_len, 3, dtype=dtype, device=device)
    sample_plan = []

    for i in range(n_segments):
        traj  = valid_trajs[
            torch.randint(len(valid_trajs), (1,), generator=generator).item()
        ]
        start = int(
            traj['valid_start_idx'][
                torch.randint(
                    traj['valid_start_idx'].numel(), (1,), generator=generator
                ).item()
            ].item()
        )
        stop = start + segment_len
        x0_seg[i]  = traj['state_traj'][start]
        u_seg[i]   = traj['u'][0, start:stop, :]
        q1_seg[i]  = traj['q1'][start:stop, :]
        sample_plan.append({'traj_id': traj['id'], 'start_idx': start})

    return x0_seg, u_seg, q1_seg, sample_plan


# ----------------------------------------------------------------------
# Parameter set construction and evaluation
# ----------------------------------------------------------------------

def _build_segment_diag_param_sets():
    """Fixed detuned parameter sets for segment-length discrimination testing."""
    params_true = torch.tensor(
        [_TRUE_PARAMS[name] for name in _PARAM_NAMES], dtype=torch.float64
    )
    idx = {name: i for i, name in enumerate(_PARAM_NAMES)}
    scales = torch.ones_like(params_true)
    for name in ('kb1', 'kb2', 'cg1', 'cg2'):
        scales[idx[name]] = 1.1
    for name in ('mh', 'Jb', 'Jh'):
        scales[idx[name]] = 0.9
    return (
        {'name': 'true',         'params': params_true},
        {'name': 'all_up_10',    'params': params_true * 1.1},
        {'name': 'all_down_10',  'params': params_true * 0.9},
        {'name': 'coupling_mix', 'params': params_true * scales},
    )


def _set_block_physical_params(block, params_vec):
    """Overwrite block log_params from a physical-values vector (_PARAM_NAMES order)."""
    with torch.no_grad():
        block.log_params.copy_(
            params_vec.to(block.log_params.device, dtype=block.log_params.dtype).log()
        )


def _eval_param_set_on_segments(block, params_vec, x0_seg, u_seg, q1_seg):
    """
    Evaluate one fixed parameter vector on the given segment batch.

    Builds G and polynomial constants from params_vec, runs simulate() directly
    (no _SimWrapper), returns per-segment MSE as a CPU tensor.
    """
    _set_block_physical_params(block, params_vec)
    with torch.no_grad():
        params = block._recover_params()
        kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d = params
        params_10 = torch.stack(
            [kb1 + kb2, cg1, cg2, cy, cb1 + cb2, mh, m1, m2, mb, Jb + Jh]
        )
        _, M1, M2, K, C = _build_matrices(params_10, block._Lb, d)
        alpha, beta, gamma, N0, N1, N2 = build_poly_constants(
            m1, m2, mb, mh, Jb, Jh, block._Lb, d
        )
        d0 = mh * (alpha * gamma - beta ** 2)
        G  = build_G_matrix(N0, d0, M1, M2, K, C)
        result = simulate(
            x0_seg, u_seg,
            G, K, C, mh, alpha, beta, gamma, N0, N1, N2,
            block._P, block._ts,
            bptt_mode='full', return_latents=False,
        )
    return (result.Y - q1_seg).pow(2).mean(dim=(1, 2)).cpu()


# ----------------------------------------------------------------------
# Cache helpers
# ----------------------------------------------------------------------

def _segment_diag_cache_path(save_dir, traj_specs):
    return os.path.join(
        save_dir,
        f'segment_len_diag_{_traj_set_tag(traj_specs)}_v{SEGMENT_DIAG_VERSION}.pt',
    )


def _segment_diag_cache_matches(cached, traj_specs, param_set_names):
    return (
        cached.get('version') == SEGMENT_DIAG_VERSION
        and tuple(cached.get('traj_specs', ())) == tuple(traj_specs)
        and tuple(cached.get('candidate_lengths_s', ())) == SEGMENT_DIAG_CANDIDATES_S
        and cached.get('n_segments') == SEGMENT_DIAG_N_SEGMENTS
        and cached.get('seed') == _BASE_SEED
        and tuple(cached.get('param_set_names', ())) == tuple(param_set_names)
    )


# ----------------------------------------------------------------------
# Result selection and display
# ----------------------------------------------------------------------

def _choose_segment_len_from_diag(results):
    """Choose the most robust segment length, breaking ties toward shorter windows."""
    best_true_rate = max(r['true_best_rate'] for r in results)
    candidates = [r for r in results if r['true_best_rate'] == best_true_rate]
    best_margin = max(r['median_margin'] for r in candidates)
    candidates = [r for r in candidates if r['median_margin'] == best_margin]
    return min(candidates, key=lambda r: r['segment_len'])['segment_len']


def _print_segment_diag_summary(diag):
    print('  segment_diag results:')
    print(f'  {"samples":>8}  {"seconds":>8}  {"true_best":>10}  {"margin":>10}')
    for r in diag['results']:
        print(
            f"  {r['segment_len']:>8}  "
            f"{r['segment_len_s']:>8.3f}  "
            f"{r['true_best_rate']:>10.3f}  "
            f"{r['median_margin']:>10.3e}"
        )
    print(f"  chosen segment_len: {diag['chosen_segment_len']} samples")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def run_segment_diag(traj_specs, traj_dir, save_dir):
    """
    Load cached segment-length diagnostic or run it once and cache the result.

    Parameters
    ----------
    traj_specs : sequence of dicts with keys 'id', 'file'
    traj_dir   : directory containing the .mat trajectory files
    save_dir   : directory for the cache file

    Returns
    -------
    int — chosen segment length in samples
    """
    param_sets      = _build_segment_diag_param_sets()
    param_set_names = [p['name'] for p in param_sets]
    cache_path      = _segment_diag_cache_path(save_dir, traj_specs)

    if os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=False)
        if _segment_diag_cache_matches(cached, traj_specs, param_set_names):
            print('  segment_diag: loaded from cache')
            _print_segment_diag_summary(cached)
            return int(cached['chosen_segment_len'])
        print('  segment_diag: cache mismatch — recomputing')

    print('  segment_diag: computing')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float64

    trajs = _load_trajs(traj_specs, traj_dir, dtype)
    for traj in trajs:
        traj['u']          = traj['u'].to(device)
        traj['q1']         = traj['q1'].to(device)
        traj['state_traj'] = traj['state_traj'].to(device)

    # rmse_baseline=1.0: param_loss is never called in the diagnostic so the
    # value has no effect on the MSE comparisons.
    block   = ParameterizedLFRBlock(RMSE_baseline=1.0).to(device)
    results = []

    for candidate_s in SEGMENT_DIAG_CANDIDATES_S:
        segment_len = int(round(candidate_s / float(_ts)))
        _attach_valid_start_idx(trajs, segment_len)
        x0_seg, u_seg, q1_seg, sample_plan = _sample_balanced_segments(
            trajs, segment_len, SEGMENT_DIAG_N_SEGMENTS, _BASE_SEED + segment_len
        )

        losses = {
            ps['name']: _eval_param_set_on_segments(
                block, ps['params'].to(device), x0_seg, u_seg, q1_seg
            )
            for ps in param_sets
        }
        all_losses     = torch.stack([losses[n] for n in param_set_names], dim=1)
        true_is_best   = all_losses.argmin(dim=1) == 0
        detuned_losses = torch.stack([losses[n] for n in param_set_names[1:]], dim=1)
        margin         = detuned_losses.min(dim=1).values - losses['true']

        results.append({
            'segment_len':      segment_len,
            'segment_len_s':    candidate_s,
            'true_best_rate':   float(true_is_best.double().mean().item()),
            'median_margin':    float(margin.median().item()),
            'mean_loss_by_set': {n: float(losses[n].mean().item()) for n in param_set_names},
            'sample_plan':      sample_plan,
        })

    chosen = _choose_segment_len_from_diag(results)
    diag = {
        'version':                   SEGMENT_DIAG_VERSION,
        'traj_specs':                tuple(traj_specs),
        'candidate_lengths_s':       SEGMENT_DIAG_CANDIDATES_S,
        'candidate_lengths_samples': [r['segment_len'] for r in results],
        'n_segments':                SEGMENT_DIAG_N_SEGMENTS,
        'seed':                      _BASE_SEED,
        'param_set_names':           param_set_names,
        'results':                   results,
        'chosen_segment_len':        chosen,
    }

    os.makedirs(save_dir, exist_ok=True)
    torch.save(diag, cache_path)
    _print_segment_diag_summary(diag)
    print(f'  segment_diag: saved to {cache_path}')
    return int(chosen)


# ----------------------------------------------------------------------
# Verification
# (run as: conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.segment_diag)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    def check(name, cond, detail=''):
        status = 'PASS' if cond else 'FAIL'
        suffix = f'  ({detail})' if detail else ''
        print(f'  {name:<55s}  {status}{suffix}')
        return cond

    print('=' * 60)
    print('segment_diag.py structural verification')
    print('=' * 60)

    results = []

    # ------------------------------------------------------------------
    # Check 1 — _traj_set_tag
    # ------------------------------------------------------------------
    print('\nCheck 1: _traj_set_tag')
    specs = [
        {'id': 'T1', 'file': 'f1.mat'},
        {'id': 'T3', 'file': 'f3.mat'},
    ]
    tag = _traj_set_tag(specs)
    results.append(check('tag is T1-T3', tag == 'T1-T3', repr(tag)))
    results.append(check('single traj tag', _traj_set_tag(specs[:1]) == 'T1'))

    # ------------------------------------------------------------------
    # Check 2 — _choose_segment_len_from_diag: tie-breaking toward shorter
    # ------------------------------------------------------------------
    print('\nCheck 2: _choose_segment_len_from_diag')
    mock_results = [
        {'segment_len': 2000, 'true_best_rate': 1.0, 'median_margin': 0.5},
        {'segment_len': 4000, 'true_best_rate': 1.0, 'median_margin': 0.5},
        {'segment_len': 8000, 'true_best_rate': 0.9, 'median_margin': 0.4},
    ]
    chosen = _choose_segment_len_from_diag(mock_results)
    results.append(check('tie broken toward shorter window',
                         chosen == 2000, f'got {chosen}'))

    mock_results_2 = [
        {'segment_len': 2000, 'true_best_rate': 0.7, 'median_margin': 0.1},
        {'segment_len': 4000, 'true_best_rate': 1.0, 'median_margin': 0.5},
    ]
    chosen2 = _choose_segment_len_from_diag(mock_results_2)
    results.append(check('higher true_best_rate preferred',
                         chosen2 == 4000, f'got {chosen2}'))

    # ------------------------------------------------------------------
    # Check 3 — _segment_diag_cache_matches
    # ------------------------------------------------------------------
    print('\nCheck 3: _segment_diag_cache_matches')
    specs_4 = [{'id': 'T1', 'file': 'f1.mat'}]
    names_4 = ['true', 'all_up_10', 'all_down_10', 'coupling_mix']
    valid_cache = {
        'version':             SEGMENT_DIAG_VERSION,
        'traj_specs':          tuple(specs_4),
        'candidate_lengths_s': SEGMENT_DIAG_CANDIDATES_S,
        'n_segments':          SEGMENT_DIAG_N_SEGMENTS,
        'seed':                _BASE_SEED,
        'param_set_names':     names_4,
    }
    results.append(check('valid cache matches', _segment_diag_cache_matches(
        valid_cache, specs_4, names_4)))
    bad_version = {**valid_cache, 'version': 0}
    results.append(check('wrong version rejects',
                         not _segment_diag_cache_matches(bad_version, specs_4, names_4)))
    bad_specs = {**valid_cache, 'traj_specs': tuple([{'id': 'T2', 'file': 'f2.mat'}])}
    results.append(check('wrong traj_specs rejects',
                         not _segment_diag_cache_matches(bad_specs, specs_4, names_4)))

    # ------------------------------------------------------------------
    # Check 4 — _build_segment_diag_param_sets: true params at index 0
    # ------------------------------------------------------------------
    print('\nCheck 4: _build_segment_diag_param_sets')
    param_sets = _build_segment_diag_param_sets()
    results.append(check('four parameter sets', len(param_sets) == 4))
    results.append(check('first set is true params', param_sets[0]['name'] == 'true'))
    results.append(check('all sets have params tensor',
                         all(isinstance(ps['params'], torch.Tensor) for ps in param_sets)))
    results.append(check('all params positive (log-safe)',
                         all(ps['params'].min().item() > 0 for ps in param_sets)))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print('=' * 60)
    print(f"Overall: {'ALL PASS' if all(results) else 'SOME FAILED'}")
    print('=' * 60)
    print()
    print('Note: run_segment_diag() requires real .mat trajectory files.')
    print('Triggered automatically by precompute() on first run.')
