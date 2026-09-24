"""Bootstrap shared by every script in this folder: vendored imports, the grey box's CFG, checks.

BB-002: `vendor/` goes first on sys.path so `model_augmentation` and `gantry_dynamic` resolve to
the vendored copies; the grey box's configuration is READ from the repo's own entry file at run
time (never retyped), with the vendored packages already in sys.modules so the entry file's
imports bind to them.
"""
import difflib
import importlib.util
import os
import sys
from dataclasses import replace

ADAPTER = os.path.dirname(os.path.abspath(__file__))
BB = os.path.dirname(ADAPTER)                                    # scripts/gantry/blackbox-closed-loop
VENDOR = os.path.join(BB, 'vendor')
REPO = os.path.abspath(os.path.join(BB, '..', '..', '..'))
ENTRY = os.path.join(REPO, 'scripts', 'gantry', 'gantry_interconnect_dynamic.py')
OUT = os.path.join(BB, 'outputs')

# The entry file's experiment overrides. Any of them set would make CFG a different experiment.
_ENTRY_ENV = ('DATASET_AB_MODE', 'DATASET_AB_SEED', 'BURN_IN_AB', 'LBFGS_AB_FREEZE', 'OBC_ARM')

# Vendored copy -> source, for check_vendor (G0 criterion 3).
_VENDOR_MAP = {
    'gantry_dynamic': os.path.join(REPO, 'scripts', 'gantry', 'gantry_dynamic'),
    'model_augmentation': os.path.join(REPO, 'model_augmentation'),
    'ann_blackbox': os.path.join(REPO, 'scripts', 'gantry', 'ann-blackbox'),
    'msd_offset': os.path.join(REPO, 'scripts', 'gantry', 'msd-offset'),
    'common': os.path.join(REPO, 'scripts', 'gantry', 'common'),
}
# Module name prefixes that must come from vendor/ (G0 criterion 1).
VENDORED_PREFIXES = ('model_augmentation', 'gantry_dynamic', 'bla_init', 'plant', 'frf_init',
                     'vector_fit', 'lpm_frf', 'data_bb_old', 'common')


def bootstrap(threads=4):
    for p in (os.path.join(VENDOR, 'ann_blackbox'), VENDOR, os.path.join(BB, 'score'), ADAPTER):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    import torch
    torch.set_num_threads(threads)          # handoff section 13
    import gantry_dynamic                   # noqa: F401  vendored; puts vendor/ on sys.path
    import model_augmentation               # noqa: F401
    import common                           # noqa: F401  vendored regular package (BB-002)
    assert os.path.abspath(common.__file__).startswith(VENDOR), common.__file__
    return torch


def load_cfg(local=True):
    """(cfg, CFG, entry_module). CFG is the entry file's object; cfg is CFG with ONLY device='cpu'
    and compile_mode=None when `local` (BB-001), else CFG itself."""
    bootstrap()
    bad = {k: os.environ[k] for k in _ENTRY_ENV if k in os.environ}
    if bad:
        raise RuntimeError('entry-file experiment overrides are set: %s; CFG would not be the '
                           "grey box's production configuration" % bad)
    spec = importlib.util.spec_from_file_location('gantry_interconnect_dynamic', ENTRY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['gantry_interconnect_dynamic'] = mod
    spec.loader.exec_module(mod)
    CFG = mod.CFG
    cfg = replace(CFG, device='cpu', compile_mode=None) if local else CFG
    return cfg, CFG, mod


def describe_cfg(cfg, CFG):
    keys = ('mode', 'fs_new', 'ts_new', 'nf', 'na_nb', 'stride', 'burn_in', 'batch_size', 'epochs',
            'its_per_val', 'n_its', 'seed', 'closed_loop', 'use_f64', 'lr', 'adam_eps', 'lbfgs',
            'device', 'compile_mode')
    lines = []
    for k in keys:
        v, V = getattr(cfg, k), getattr(CFG, k)
        lines.append('  %-13s %r%s' % (k, v, '' if v == V else '   (CFG: %r, local override)' % (V,)))
    return '\n'.join(lines)


def check_modules():
    """G0 criterion 1: every loaded module with a vendored prefix comes from vendor/."""
    bad, n = [], 0
    for name, m in list(sys.modules.items()):
        if not name.split('.')[0] in VENDORED_PREFIXES:
            continue
        f = getattr(m, '__file__', None)
        if f is None:
            continue
        n += 1
        if not os.path.abspath(f).startswith(VENDOR):
            bad.append((name, f))
    return n, bad


def check_vendor():
    """G0 criterion 3: each vendored file equals its source except hunks carrying VENDOR-PATCH."""
    report = []
    for sub, src_root in _VENDOR_MAP.items():
        root = os.path.join(VENDOR, sub)
        for dp, _, fs in os.walk(root):
            for f in fs:
                if not f.endswith('.py'):
                    continue
                cp = os.path.join(dp, f)
                sp = os.path.join(src_root, os.path.relpath(cp, root))
                b = open(cp, encoding='utf-8').read().splitlines()
                if not os.path.exists(sp):
                    # vendor-only file: allowed only if it declares itself as a patch
                    marked = any('VENDOR-PATCH' in ln for ln in b)
                    report.append((os.path.relpath(cp, VENDOR) + ' (vendor-only)', 1, 0 if marked else 1))
                    continue
                a = open(sp, encoding='utf-8').read().splitlines()
                hunks = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
                         if op[0] != 'equal']
                unmarked = [op for op in hunks if not any('VENDOR-PATCH' in ln for ln in b[op[3]:op[4]])]
                report.append((os.path.relpath(cp, VENDOR), len(hunks), len(unmarked)))
    return report
