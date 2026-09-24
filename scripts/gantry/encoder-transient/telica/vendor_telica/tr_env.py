"""Import bootstrap for every telica-real script (TR-001).

Usage at the top of a script in a telica-real subfolder:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import tr_env                      # noqa: E402  (paths, threads, leak guard)

Puts `telica-real/vendor` (vendored framework) and `telica-real/` (this folder's own packages)
on sys.path, pins torch/OMP threads to 4, and offers `check_no_leak()`, which fails if any
project module was imported from outside telica-real. The repo is pip-installed in develop mode
(`ModelAugmentation.egg-info`), so a script that skipped this bootstrap would silently import
the ORIGINAL `model_augmentation`; the leak check is what catches that.
"""
import os
import sys

TR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TR, '..', '..', '..', '..', '..'))   # VENDORED: repo root
VENDOR = os.path.join(TR, 'vendor')
OUTPUTS = os.path.abspath(os.path.join(TR, '..', '..', 'outputs', 'telica'))   # VENDORED

for _p in (TR, VENDOR):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)          # VENDOR ends up first

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
try:
    import torch
    torch.set_num_threads(4)
except ImportError:                  # pragma: no cover
    pass

_OWN = ('model_augmentation', 'gantry_dynamic', 'common', 'real_data_verification',
        'params', 'friction', 'controller', 'baseline', 'learnability', 'pipeline')


def check_no_leak():
    """Raise if a project module resolved outside telica-real."""
    leaks = [n for n, m in list(sys.modules.items())
             if getattr(m, '__file__', None) and n.split('.')[0] in _OWN
             and not os.path.abspath(m.__file__).startswith(TR)]
    if leaks:
        raise ImportError('modules imported from outside telica-real: %s' % leaks)


def out_dir(run):
    """telica-real/outputs/<run>, created. The only place scripts write artefacts."""
    d = os.path.join(OUTPUTS, run)
    os.makedirs(d, exist_ok=True)
    return d
