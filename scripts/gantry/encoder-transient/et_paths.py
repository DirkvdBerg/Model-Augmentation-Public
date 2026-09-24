"""sys.path for every encoder-transient script: the vendored copies first, never the repo's (ET-001).

Import this before anything from gantry_dynamic / model_augmentation. `check()` asserts that the
modules actually resolved inside vendor/, so a stray repo path cannot silently swap the code.
"""
import os
import sys

ET = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(ET, 'vendor')
REPO = os.path.abspath(os.path.join(ET, '..', '..', '..'))
OUT = os.path.join(ET, 'outputs')
MODE = 'augmentation_ma50_b140-230_a6_z03'

for p in (os.path.join(VENDOR, 'closed_loop'), os.path.join(VENDOR, 'transient'), VENDOR, ET):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
# The repo root and scripts/gantry must not shadow the vendored packages.
for p in (REPO, os.path.join(REPO, 'scripts', 'gantry')):
    while p in sys.path:
        sys.path.remove(p)


def check(verbose=True):
    import gantry_dynamic
    import model_augmentation
    import truth
    for m in (gantry_dynamic, model_augmentation, truth):
        f = os.path.abspath(m.__file__)
        assert f.startswith(VENDOR), f'{m.__name__} resolved outside vendor/: {f}'
        if verbose:
            print(f'  [et_paths] {m.__name__:<20} {os.path.relpath(f, ET)}')
    return True
