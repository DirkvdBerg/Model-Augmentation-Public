"""Shared part of the clean-code equivalence harness (tasks/clean-code/design.md section 6).

Every harness process imports this module FIRST: it pins the thread settings before torch or numpy
load (6.2), puts exactly one code root on sys.path (mainline: scripts/gantry, with
model_augmentation from the repository root; clean: Clean/ alone) and builds the run configuration
of a runs.tsv row the way that side builds it. Mainline is imported unmodified (6.2): the THESIS_*
variables of the row plus THESIS_SMOKE=1 (the PS2 smoke spec, training.py:875) and
THESIS_DEVICE=cpu, then EP.CFG is replaced by the harness budget.
"""
import os
import sys

# HARNESS_THREADS: only the GPU smokes of 6.4 (not bitwise) may raise it; every CPU layer runs at 1
N_THREADS = int(os.environ.get('HARNESS_THREADS', '1'))
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = str(N_THREADS)

import hashlib
import json
import importlib.metadata as _md
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
CLEAN = os.path.join(REPO, 'Clean')
GOLDEN = os.path.join(REPO, 'simulations', 'clean-golden')
CLEAN_OUT = os.path.join(GOLDEN, '_clean')

# runs.tsv rows 221 to 230 (no 228): preset, arm, start, n_a, ps2 (design section 3, Presets)
ROWS = {
    221: ('R1', 'obc', 'detuned', 2, 1), 222: ('R2', 'obc', 'correct', 2, 1),
    223: ('R3', 'u', 'correct', 2, 0), 224: ('R4', 'u', 'detuned', 2, 1),
    225: ('R5', 'u', 'correct', 2, 1), 226: ('R6', 'u', 'correct', 0, 0),
    227: ('R7', 'u', 'correct', 8, 1), 229: ('R13', 'obc', 'detuned', 0, 0),
    230: ('R14', 'u', 'detuned', 0, 0),
}
# design 6.2: identical on both sides
BUDGET = dict(truncate_s=0.6, epochs=3, n_its=None, its_per_val=5,
              lbfgs_max_iter=2, lbfgs_inner_iter=1, lbfgs_windows=256)

# the 19 copied mainline files (design 6.2 fingerprint), mainline path -> clean path
COPIED = {
    'scripts/gantry/gantry_interconnect_dynamic.py': 'gantry_interconnect_dynamic.py',
    'scripts/gantry/gantry_dynamic/config.py': 'gantry_dynamic/config.py',
    'scripts/gantry/gantry_dynamic/data.py': 'gantry_dynamic/data.py',
    'scripts/gantry/gantry_dynamic/controller.py': 'gantry_dynamic/controller.py',
    'scripts/gantry/gantry_dynamic/model.py': 'gantry_dynamic/model.py',
    'scripts/gantry/gantry_dynamic/obc_gantry.py': 'gantry_dynamic/obc_gantry.py',
    'scripts/gantry/gantry_dynamic/training.py': 'gantry_dynamic/training.py',
    'scripts/gantry/gantry_dynamic/closed_loop_report.py': 'gantry_dynamic/closed_loop_report.py',
    'model_augmentation/fit_systems/interconnect.py': 'model_augmentation/fit_systems/interconnect.py',
    'model_augmentation/fit_systems/blocks.py': 'model_augmentation/fit_systems/blocks.py',
    'model_augmentation/fit_systems/closed_loop.py': 'model_augmentation/fit_systems/closed_loop.py',
    'model_augmentation/fit_systems/lbfgs_polish.py': 'model_augmentation/fit_systems/lbfgs_polish.py',
    'model_augmentation/fit_systems/obc.py': 'model_augmentation/fit_systems/obc.py',
    'model_augmentation/fit_systems/pre_encoder.py': 'model_augmentation/fit_systems/pre_encoder.py',
    'model_augmentation/fit_systems/ps2_opening.py': 'model_augmentation/fit_systems/ps2_opening.py',
    'model_augmentation/systems/gantry_ss.py': 'model_augmentation/systems/gantry_ss.py',
    'model_augmentation/utils/utils.py': 'model_augmentation/utils/utils.py',
    'model_augmentation/utils/torch_nets.py': 'model_augmentation/utils/torch_nets.py',
    'model_augmentation/utils/deepSI_corrections.py': 'model_augmentation/utils/deepSI_corrections.py',
}

SIDE = None
CLEAN_AT_START = None


def select_side(side):
    """Put the one code root of `side` on sys.path. Call before any pipeline import."""
    global SIDE
    if side not in ('mainline', 'clean'):
        raise ValueError(side)
    SIDE = side
    global CLEAN_AT_START
    CLEAN_AT_START = clean_tree_hashes()     # H2: the tree this process imports
    if side == 'mainline':
        sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))
    else:
        sys.path.insert(0, CLEAN)
    import torch
    torch.set_num_threads(N_THREADS)


def assert_roots():
    """Every loaded gantry_dynamic* / model_augmentation* module lies under the selected root (6.2)."""
    bad = []
    for name, mod in list(sys.modules.items()):
        if not (name.split('.')[0] in ('gantry_dynamic', 'model_augmentation')):
            continue
        f = getattr(mod, '__file__', None)
        if f is None:
            continue
        f = os.path.abspath(f)
        if SIDE == 'clean':
            ok = f.startswith(CLEAN + os.sep)
        else:
            ok = ((f.startswith(os.path.join(REPO, 'scripts', 'gantry') + os.sep)
                   or f.startswith(os.path.join(REPO, 'model_augmentation') + os.sep))
                  and not f.startswith(CLEAN + os.sep))
        if not ok:
            bad.append((name, f))
    if bad:
        raise RuntimeError('modules outside the %s root: %s' % (SIDE, bad))
    return True


def make_cfg(row, noise, run_id, budget=None, **extra):
    """The run configuration of `row` on `noise` for the selected side, with the harness budget."""
    preset, arm, start, na, ps2 = ROWS[row]
    budget = dict(BUDGET if budget is None else budget)
    os.environ['SLURM_JOB_ID'] = str(run_id)
    if SIDE == 'mainline':
        os.environ.update(THESIS_ARM=arm, THESIS_NOISE=noise, THESIS_START=start, THESIS_NA=str(na),
                          THESIS_SEED='1', THESIS_NF='0.1', THESIS_ARCH='mlp', THESIS_PS2=str(ps2),
                          THESIS_INIT='live', THESIS_SMOKE='1', THESIS_DEVICE='cpu')
        import gantry_interconnect_dynamic as EP
        cfg = replace(EP.CFG, **budget, **extra)
        EP.CFG = cfg
        return cfg, EP
    from gantry_dynamic.config import RunConfig, PRESETS
    p = PRESETS[preset]
    assert (p['arm'], p['start'], p['n_a'], int(p['ps2'])) == (arm, start, na, ps2), (preset, p)
    kw = dict(device='cpu', compile_mode=None, ps2_smoke=True)
    kw.update(budget)
    kw.update(extra)
    cfg = RunConfig(**p, noise=noise, **kw)
    import gantry_interconnect_dynamic as EP
    return cfg, EP


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def clean_tree_hashes():
    """H2: SHA256 of every file under Clean/ (package initialisers and the import-origin guard
    included, __pycache__ excluded), keyed by the path relative to Clean/ with '/' separators."""
    out = {}
    for root, dirs, files in os.walk(CLEAN):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        for f in sorted(files):
            p = os.path.join(root, f)
            out[os.path.relpath(p, CLEAN).replace(os.sep, '/')] = sha256_file(p)
    return out


def clean_tree_problems(recorded):
    """H2: [] when `recorded`, the clean manifest an output was made with, equals the current clean
    tree; else every file added, removed or changed since. A clean output made from another tree
    is refused, never compared."""
    if not recorded:
        return ['the clean output records no clean source manifest']
    now = clean_tree_hashes()
    return ['clean source changed since the clean output was made: %s' % k
            for k in sorted(set(now) | set(recorded)) if now.get(k) != recorded.get(k)]


def source_hashes():
    """SHA256 of the 19 copied mainline files, and of the full clean tree (H2)."""
    return {'mainline': {k: sha256_file(os.path.join(REPO, k)) for k in COPIED},
            'clean': clean_tree_hashes()}


def data_hashes(data_dir, files):
    """SHA256 of both MANIFEST.csv files and of every record `files` names (below data_dir)."""
    root = os.path.dirname(data_dir)
    out = {'MANIFEST:' + d: sha256_file(os.path.join(root, d, 'MANIFEST.csv'))
           for d in ('Coulomb-tanh-and-MSD-lowpass-noise', 'Coulomb-tanh-and-MSD-noise-free')}
    for f in files:
        out[f] = sha256_file(os.path.join(data_dir, f))
    return out


def lib_versions():
    import numpy, scipy, torch
    return {'deepSI': _md.version('deepSI'), 'torch': torch.__version__, 'numpy': numpy.__version__,
            'scipy': scipy.__version__, 'python': sys.version.split()[0]}


def thread_settings():
    import torch
    return {'torch_num_threads': torch.get_num_threads(),
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
            'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS')}


def write_meta(out_dir, cfg, data, extra=None):
    """meta.json of design 6.2: sources (the clean manifest is the full tree, H2), manifests,
    records, config, data root, libraries, threads, git provenance, budget."""
    from gantry_dynamic.config import git_provenance, config_json_dict
    from gantry_dynamic.data import traj_dir
    ddir = traj_dir(cfg)
    files = list(data.train_files) + list(data.val_files) + list(data.test_files)
    try:
        cj = json.loads(json.dumps(config_json_dict(cfg, git=None), default=str))
    except Exception as e:                   # informational only
        cj = 'unavailable: %r' % (e,)
    sources = source_hashes()
    if sources['clean'] != CLEAN_AT_START:
        # the clean tree changed while this process ran: record no manifest, so H2 refuses the output
        sources['clean'] = {}
    meta = dict(side=SIDE, sources=sources, data=data_hashes(ddir, files),
                data_dir=ddir, files=dict(train=list(data.train_files), val=list(data.val_files),
                                          test=list(data.test_files)),
                config=cj, libraries=lib_versions(), threads=thread_settings(),
                git=git_provenance(), budget=BUDGET)
    if extra:
        meta.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1, default=str)
    return meta


# array capture
HASH_ABOVE_BYTES = 8 << 20      # larger arrays are stored as SHA256 + summary unless --full


class Capture:
    """name -> array; arrays above HASH_ABOVE_BYTES keep only a SHA256 of their bytes (full=False)."""

    def __init__(self, full=False):
        self.arrays, self.hashed, self.info, self.full = {}, {}, {}, full

    def add(self, name, a):
        import numpy as np
        import torch
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        a = np.asarray(a)
        if a.dtype == object:
            raise TypeError('%s is an object array' % name)
        a = np.ascontiguousarray(a)
        if self.full or a.nbytes <= HASH_ABOVE_BYTES:
            self.arrays[name] = a
        else:
            fin = a[np.isfinite(a)] if a.dtype.kind == 'f' else a
            self.hashed[name] = dict(sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                                     shape=list(a.shape), dtype=str(a.dtype),
                                     sum=float(fin.sum()) if fin.size else 0.0,
                                     absmax=float(abs(fin).max()) if fin.size else 0.0)

    def add_state(self, prefix, sd):
        self.info[prefix + ':keys'] = list(sd.keys())
        self.info[prefix + ':shapes'] = [list(v.shape) for v in sd.values()]
        for k, v in sd.items():
            self.add('%s/%s' % (prefix, k), v)

    def save(self, out_dir, stem):
        import numpy as np
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, stem + '.npz'), **self.arrays)
        with open(os.path.join(out_dir, stem + '.json'), 'w') as fh:
            json.dump(dict(hashed=self.hashed, info=self.info), fh, indent=1, default=str)
