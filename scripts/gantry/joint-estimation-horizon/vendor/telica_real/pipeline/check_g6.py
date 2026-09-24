"""G6 static checks of the server configuration (no data, no training).

Builds the SERVER RunConfig exactly as telica_augment.make_cfg does and checks the handoff's
section-4 items that are configuration, plus the refusals that keep the simulation defaults out.
"""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402

os.environ.pop('TELICA_SMOKE', None)
from pipeline import telica_augment as ta                              # noqa: E402
from pipeline.real_data import NORM_FILE, load_datasets_real           # noqa: E402
from controller.telica_bank import build_telica_bank                   # noqa: E402
tr_env.check_no_leak()

ok_all = True


def check(label, ok, detail=''):
    global ok_all
    ok_all &= bool(ok)
    print(f'{"PASS" if ok else "FAIL"}  {label}  {detail}')


cfg = ta.make_cfg()
FS_SET = int(ta.ann_settings().get('fs_train', 20000))
check('training rate = ann_settings fs_train, 20 kHz or 5 kHz only (item 4, TR-008/TR-022)',
      abs(1.0 / cfg.ts_new - FS_SET) < 1e-6 and FS_SET in (20000, 5000),
      f'fs_train={FS_SET} ts_new={cfg.ts_new}')
check('float64 (TR-013)', cfg.use_f64)
AS = ta.ann_settings()
check('ANN settings from ann_settings.json (item 7; TR-018/TR-024)',
      cfg.nx_ann == int(AS['nx_ann']) and tuple(cfg.ann_route_ix) == tuple(AS['ann_route_ix'])
      and tuple(cfg.ann_route_ix) == tuple([3, 4, 5] + list(range(6, 6 + cfg.nx_ann)))
      and cfg.nf == int(round(float(AS['nf_seconds']) * FS_SET)),
      f'nx_ann={cfg.nx_ann} route={cfg.ann_route_ix} nf={cfg.nf} na_nb={cfg.na_nb}')
check('closed loop, GPU, compiled, chunked', cfg.closed_loop and cfg.device == 'cuda'
      and cfg.compile_mode == 'reduce-overhead'
      and cfg.checkpoint_chunk == int(ta.ann_settings()['checkpoint_chunk']),
      f'chunk={cfg.checkpoint_chunk} batch={cfg.batch_size} epochs={cfg.epochs}')
check('no OBC / orth / joint estimation / zero-mean penalty', not (cfg.obc or cfg.orth
                                                                    or cfg.joint_estimation))
check('frozen normalisation present (item 3)', os.path.isfile(NORM_FILE))
try:
    load_datasets_real(replace(cfg, fs_new=4000), max_records=1)
    refused = False
except ValueError:
    refused = True
check('4 kHz decimation refused by the data layer (item 4)', refused)
try:
    build_telica_bank(1 / 4000, ystd=np.ones(3), std_u=np.ones(3))
    refused = False
except ValueError:
    refused = True
check('Telica bank refuses a 4 kHz step (no controller at that rate; item 1, TR-008/TR-022)', refused)
import ast                                                             # noqa: E402


def code_names(path):
    """Imported modules and referenced names/attributes of a module, docstrings excluded."""
    tree = ast.parse(open(path, encoding='utf-8').read())
    mods, names = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module); names.update(a.name for a in n.names)
        elif isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
    return mods, names


mods, names = code_names(ta.__file__)
check('entry point never imports the simulation data layer or ruleOfThumb bank (items 1, 6)',
      not any(m in mods for m in ('gantry_dynamic.data', 'gantry_dynamic.controller')), str(sorted(mods)))
GT = ('load_mat_aug', 'load_datasets', 'state_recovery_diagnostic', 'compute_all_baselines',
      'oracle_open_loop', 'evaluate_and_save')
check('no ground-truth diagnostics called by the entry point (item 5)',
      not any(g in names for g in GT), str([g for g in GT if g in names]))
import gantry_dynamic.training as gtr                                  # noqa: E402
tsrc = open(gtr.__file__, encoding='utf-8').read()
check('training wrapper skips the absorber-GT diagnostic on real data (item 5)',
      "getattr(data, 'val_x_aug', None) is None" in tsrc)
print('G6 STATIC', 'PASS' if ok_all else 'FAIL')
