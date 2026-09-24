"""Environment for every script in this folder (JH-001).

Import this FIRST. It puts `vendor/` at the front of sys.path so `model_augmentation`,
`gantry_dynamic` and `common` resolve to the vendored copies, and `check_no_leak()` proves it:
the repo is pip-installed in develop mode, so an unguarded import can silently reach the live
pipeline instead.
"""
import os
import sys

JH = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(JH, 'vendor')
REPO = os.path.abspath(os.path.join(JH, '..', '..', '..'))
OUT = os.path.join(JH, 'outputs')
DATA = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory')

if sys.path[0] != VENDOR:
    while VENDOR in sys.path:
        sys.path.remove(VENDOR)
    sys.path.insert(0, VENDOR)

_GUARDED = ('model_augmentation', 'gantry_dynamic', 'common', 'gantry_interconnect_dynamic')


def check_no_leak(verbose=True):
    """Every loaded module of the guarded packages must live inside vendor/. Raises otherwise."""
    bad, n = [], 0
    for name, mod in list(sys.modules.items()):
        if name.split('.')[0] not in _GUARDED:
            continue
        f = getattr(mod, '__file__', None)
        if f is None:
            continue
        n += 1
        if not os.path.abspath(f).startswith(VENDOR + os.sep):
            bad.append((name, f))
    if bad:
        raise RuntimeError('vendored-import leak: %s' % bad)
    if verbose:
        print(f'[jh_env] check_no_leak: {n} guarded modules, all inside vendor/')
    return n


def out_dir(run):
    d = os.path.join(OUT, run)
    os.makedirs(d, exist_ok=True)
    return d
