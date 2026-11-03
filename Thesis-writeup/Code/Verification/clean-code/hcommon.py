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
# HARNESS_CLEAN_ROOT: another copy of Clean/ as the clean code root (D-295 arm B imports a copy of the final tree
# while arm A runs on the repository's reverted Clean/); unset: the repository's Clean/
CLEAN = os.path.abspath(os.environ.get('HARNESS_CLEAN_ROOT') or os.path.join(REPO, 'Clean'))
# HARNESS_GOLDEN: the root of the layer 1 to 3c goldens and clean outputs (server_suite.sbatch captures both sides
# afresh in its own root, since the comparators refuse goldens of another data folder or library set); unset: the
# repository's goldens. The snapshots stay under simulations/clean-golden/src-snapshots either way (SNAPSHOTS).
GOLDEN = os.environ.get('HARNESS_GOLDEN') or os.path.join(REPO, 'simulations', 'clean-golden')
SNAPSHOTS = os.path.join(REPO, 'simulations', 'clean-golden', 'src-snapshots')
# HARNESS_CLEAN_OUT: the root of the clean outputs; unset: GOLDEN/_clean. D-294 arm B captures the final tree into
# its own root beside arm A's (compare_d294.py), against the same goldens.
CLEAN_OUT = os.environ.get('HARNESS_CLEAN_OUT') or os.path.join(GOLDEN, '_clean')

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
    # D-292: no presets in Clean/; the row's axes over the entry script's SETTINGS, with the short PS opening
    from gantry_dynamic.config import RunConfig
    from gantry_dynamic.data import DATASETS
    import gantry_interconnect_dynamic as EP
    noise = clean_noise(noise, DATASETS)
    kw = dict(device='cpu', compile_mode=None, ps2_opening=short_ps2_opening(EP.SETTINGS))
    kw.update(budget)
    kw.update(extra)
    clean_inner_iter(kw)
    cfg = RunConfig(**dict(EP.SETTINGS, arm=arm, start=start, n_a=na, ps2=bool(ps2), noise=noise, **kw))
    return cfg, EP


def clean_noise(noise, datasets):
    """The clean noise setting of a harness case's noise label: the cases keep the thesis label 'lowpass' (golden
    folders, both sides), which Clean names 'noisy'; a snapshot from before that rename keeps 'lowpass'."""
    return 'noisy' if noise == 'lowpass' and 'noisy' in datasets else noise


def clean_inner_iter(kw):
    """D-292: the clean polish's iterations per call is the constant lbfgs_polish.INNER_ITER, not a RunConfig field;
    a budget's lbfgs_inner_iter leaves kw and sets that constant, as the mainline field does on its side."""
    if 'lbfgs_inner_iter' in kw:
        from model_augmentation.fit_systems import lbfgs_polish as LP
        LP.INNER_ITER = kw.pop('lbfgs_inner_iter')
    return kw


# the nine keys of mainline's cfg.hp, which the clean code no longer has (D-292); recorded for the L1 comparison
HP_KEYS = ('n_a', 'n_nodes_per_layer', 'n_hidden_layers', 'up_sample', 'nf', 'na_nb', 'batch_size', 'lr', 'epochs')


def hp_of(cfg):
    """The hp dict of cfg: mainline's own, or the same nine values read from the clean RunConfig."""
    return cfg.hp if SIDE == 'mainline' else {k: getattr(cfg, k) for k in HP_KEYS}


def train_data(data):
    """The training records as one System_data_list: mainline's data.train_data; Clean keeps the list only and its
    callers wrap it at the deepSI call."""
    import deepSI
    return data.train_data if SIDE == 'mainline' else deepSI.System_data_list(data.train_list)


def val_data(data):
    """The validation records as one System_data_list: mainline's data.val_ckpt_data, or Clean's list wrapped."""
    import deepSI
    return data.val_ckpt_data if SIDE == 'mainline' else deepSI.System_data_list(data.val_list)


def build_model(GDM, cfg, data, norm):
    """GDM.build_model of the selected side: mainline takes hp first, the clean one reads cfg only (D-292)."""
    return GDM.build_model(cfg.hp, cfg, data, norm) if SIDE == 'mainline' else GDM.build_model(cfg, data, norm)


def dropped_state_keys():
    """R3 (D-284 amendment 10): last name components of the hfn state_dict keys of the baseline block
    (connected_blocks.0) that mainline registers and the clean code does not; their one source is
    clean-up-verification/convert_run.py, which drops them from converted checkpoints."""
    if os.path.join(REPO, 'clean-up-verification') not in sys.path:
        sys.path.append(os.path.join(REPO, 'clean-up-verification'))
    from convert_run import DROPPED_STATE_KEYS
    return DROPPED_STATE_KEYS


# hfn state_dict keys mainline registers outside the baseline block and Clean does not: the OBC row selection E
# (2026-11-03: Clean pads f_obc with zeros instead)
DROPPED_OBC_KEYS = ('obc_correction.E',)


def is_dropped_state_key(key):
    """True for an hfn state_dict key that R3 dropped from the clean baseline block, or a dropped OBC key."""
    return ((key.startswith('connected_blocks.0.') and key.rsplit('.', 1)[1] in dropped_state_keys())
            or key in DROPPED_OBC_KEYS)


# OBC renames in Clean (2026-11-03): mainline hfn state_dict key -> the clean key of the same buffer
RENAMED_STATE_KEYS = {'obc_correction.vbar': 'obc_correction.beta_lin',
                      'obc_correction.theta_frozen': 'obc_correction.theta_aux'}


def clean_state_key(key):
    """The clean name of a mainline hfn state_dict key, or of a capture name ending in one; others unchanged."""
    for old, new in RENAMED_STATE_KEYS.items():
        if key == old or key.endswith(('/' + old, '.' + old)):
            return key[:-len(old)] + new
    return key


def drop_state(sd):
    """A mainline hfn state_dict without the keys R3 dropped and with the OBC keys under their clean names, so the
    clean model loads a mainline state."""
    return {clean_state_key(k): v for k, v in sd.items() if not is_dropped_state_key(k)}


def short_ps2_opening(settings):
    """E2: the clean side's short PS2 opening (formerly RunConfig.ps2_smoke), as the ps2_opening setting
    (D-292); its one source is clean-up-verification/smoke.py."""
    if os.path.join(REPO, 'clean-up-verification') not in sys.path:
        sys.path.append(os.path.join(REPO, 'clean-up-verification'))
    from smoke import short_ps2_opening as _short
    return _short(settings)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


GREY_BOX = ('gantry_interconnect_dynamic.py', 'gantry_dynamic', 'model_augmentation')


def grey_box_files():
    """The 24 grey-box .py files under Clean/ (package initialisers included, __pycache__
    excluded), as paths relative to Clean/ with '/' separators. Clean/blackbox/ and
    Clean/data_generation/ belong to other sessions and are left out (D-284 amendment 5)."""
    out = []
    for top in GREY_BOX:
        p = os.path.join(CLEAN, top)
        if os.path.isfile(p):
            out.append(top)
            continue
        for root, dirs, files in os.walk(p):
            dirs[:] = sorted(d for d in dirs if d != '__pycache__')
            out += [os.path.relpath(os.path.join(root, f), CLEAN).replace(os.sep, '/')
                    for f in sorted(files) if f.endswith('.py')]
    return sorted(out)


def clean_tree_hashes():
    """H2: SHA256 of every grey-box .py file under Clean/, keyed by its path relative to Clean/."""
    return {k: sha256_file(os.path.join(CLEAN, *k.split('/'))) for k in grey_box_files()}


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
    """SHA256 of the 19 copied mainline files, and of the grey-box clean files (H2)."""
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


def output_hashes(out_dir, skip=('meta.json',)):
    """{relative path ('/' separators): SHA256} of every file below out_dir except `skip`."""
    out = {}
    for root, dirs, files in os.walk(out_dir):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, out_dir).replace(os.sep, '/')
            if rel not in skip:
                out[rel] = sha256_file(p)
    return out


def output_problems(out_dir, meta, legacy_newer=()):
    """[] when every output `meta` lists exists below out_dir with its recorded SHA256, else the
    problems: a case whose meta.json does not describe the files next to it is refused, never
    compared. A meta.json from before the output manifest has no 'outputs'; then each file of
    `legacy_newer` must exist and be no older than meta.json (the old capture order wrote meta.json
    first, so an older file is one the capture never rewrote)."""
    if 'outputs' in meta:
        probs = []
        for rel, h in sorted(meta['outputs'].items()):
            p = os.path.join(out_dir, *rel.split('/'))
            if not os.path.exists(p):
                probs.append('output listed in meta.json is missing: %s' % p)
            elif sha256_file(p) != h:
                probs.append('output differs from its meta.json hash: %s' % p)
        return probs or ([] if meta['outputs'] else ['meta.json lists no outputs: %s' % out_dir])
    t_meta = os.path.getmtime(os.path.join(out_dir, 'meta.json'))
    probs = []
    for f in legacy_newer:
        p = os.path.join(out_dir, f)
        if not os.path.exists(p):
            probs.append('output missing: %s' % p)
        elif os.path.getmtime(p) < t_meta:
            probs.append('output older than its meta.json (stale): %s' % p)
    return probs


def write_meta(out_dir, cfg, data, extra=None):
    """meta.json of design 6.2: sources (the clean manifest is the grey-box files, H2), manifests,
    records, config, data root, libraries, threads, budget, and `outputs`, the
    SHA256 of every file already in out_dir. Call it LAST, once every output of the case is saved:
    a meta.json then exists only for a complete case and pins the files it was written for."""
    from gantry_dynamic.config import config_json_dict
    import gantry_dynamic.data as GDD
    # the data folder of the run: traj_dir in mainline, dataset_dir in Clean
    ddir = (GDD.traj_dir if SIDE == 'mainline' else GDD.dataset_dir)(cfg)
    files = list(data.train_files) + list(data.val_files) + list(data.test_files)
    try:
        cj = json.loads(json.dumps(config_json_dict(cfg), default=str))
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
                budget=BUDGET)
    if extra:
        meta.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    meta['outputs'] = output_hashes(out_dir)
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
