"""Step 3 of the handoff: import every module of Clean/ with only Clean/ on the path.

    python -I -u check_imports.py

-I keeps the script folder and the user site off sys.path; Clean/ is then inserted first, exactly
as `python Clean/gantry_interconnect_dynamic.py` does. Every loaded gantry_dynamic* and
model_augmentation* module must lie under Clean/.
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'Clean'))
sys.path.insert(0, CLEAN)

MODULES = [
    'gantry_interconnect_dynamic',
    'gantry_dynamic', 'gantry_dynamic.config', 'gantry_dynamic.data', 'gantry_dynamic.controller',
    'gantry_dynamic.model', 'gantry_dynamic.obc_gantry', 'gantry_dynamic.training',
    'gantry_dynamic.closed_loop_report',
    'model_augmentation', 'model_augmentation.fit_systems', 'model_augmentation.fit_systems.interconnect',
    'model_augmentation.fit_systems.blocks', 'model_augmentation.fit_systems.closed_loop',
    'model_augmentation.fit_systems.lbfgs_polish', 'model_augmentation.fit_systems.obc',
    'model_augmentation.fit_systems.pre_encoder', 'model_augmentation.fit_systems.ps2_opening',
    'model_augmentation.systems', 'model_augmentation.systems.gantry_ss',
    'model_augmentation.utils', 'model_augmentation.utils.utils', 'model_augmentation.utils.torch_nets',
    'model_augmentation.utils.deepSI_corrections',
]
for m in MODULES:
    importlib.import_module(m)
bad = [(n, getattr(mod, '__file__', None)) for n, mod in sys.modules.items()
       if n.split('.')[0] in ('gantry_dynamic', 'model_augmentation', 'gantry_interconnect_dynamic')
       and not os.path.abspath(mod.__file__).startswith(CLEAN + os.sep)]
# grey box only: Clean/blackbox/ and Clean/data_generation/ belong to other sessions (D-284 amendment 5)
n_files = sum(1 for top in ('gantry_dynamic', 'model_augmentation')
              for r, _, fs in os.walk(os.path.join(CLEAN, top)) for f in fs if f.endswith('.py')) + 1
print('imported %d modules; %d grey-box .py files under Clean/; outside Clean/: %s'
      % (len(MODULES), n_files, bad or 'none'))
sys.exit(1 if bad or n_files != len(MODULES) else 0)
