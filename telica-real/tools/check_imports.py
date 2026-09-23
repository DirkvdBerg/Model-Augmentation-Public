"""G0: every module on the telica-real path resolves INSIDE telica-real (TR-001)."""
import os
import sys

TR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TR, 'vendor'))
sys.path.insert(0, TR)

import importlib

MODS = [
    'model_augmentation.systems.gantry_ss',
    'model_augmentation.fit_systems.blocks',
    'model_augmentation.fit_systems.closed_loop',
    'model_augmentation.fit_systems.interconnect',
    'gantry_dynamic.config',
    'gantry_dynamic.controller',
    'gantry_dynamic.data',
    'gantry_dynamic.model',
    'real_data_verification.telica_loader',
    'real_data_verification.telica_controller',
]
extra = sys.argv[1:]
bad = 0
for name in MODS + extra:
    m = importlib.import_module(name)
    f = os.path.abspath(m.__file__)
    ok = f.startswith(TR)
    bad += (not ok)
    print(f'{"OK " if ok else "BAD"} {name:45s} {os.path.relpath(f, os.path.dirname(TR))}')
# Any module from the repo's own packages loaded from outside telica-real is a leak.
leaks = [n for n, m in sys.modules.items()
         if getattr(m, '__file__', None)
         and (n.split('.')[0] in ('model_augmentation', 'gantry_dynamic', 'common',
                                  'real_data_verification', 'params', 'friction',
                                  'controller', 'baseline', 'learnability', 'pipeline'))
         and not os.path.abspath(m.__file__).startswith(TR)]
print('leaks:', leaks)
print('IMPORTS PASS' if bad == 0 and not leaks else 'IMPORTS FAIL')
