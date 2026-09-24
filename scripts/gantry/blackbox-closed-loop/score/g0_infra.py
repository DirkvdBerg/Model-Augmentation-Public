"""G0: vendored imports resolve inside the folder; CFG is the entry file's; vendor == source (BB-002)."""
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'adapter'))
import bbcl                                                                # noqa: E402

cfg, CFG, entry = bbcl.load_cfg(local=True)
import bb_model                                                            # noqa: E402,F401
import cl_score                                                            # noqa: E402,F401
import bla_init                                                            # noqa: E402,F401
import gantry_dynamic.model, gantry_dynamic.controller, gantry_dynamic.training   # noqa: E402,F401,E401
from gantry_dynamic.data import traj_dir, TRAIN_FILES, VAL_FILES, TEST_FILES     # noqa: E402
from model_augmentation.fit_systems import closed_loop                     # noqa: E402

# The old-data loader of the four checkpoints (G2), imported by path under a non-clashing name.
_spec = importlib.util.spec_from_file_location(
    'data_bb_old', os.path.join(bbcl.VENDOR, 'ann_blackbox', 'data.py'))
data_bb_old = importlib.util.module_from_spec(_spec)
sys.modules['data_bb_old'] = data_bb_old
_spec.loader.exec_module(data_bb_old)
import plant                                                               # noqa: E402

res = {}
n, bad = bbcl.check_modules()
res['c1_modules_checked'] = n
res['c1_not_vendored'] = bad
print('[G0.1] %d vendored-prefix modules loaded, %d from outside vendor/ %s' % (n, len(bad), bad))
print('       closed_loop.py  -> %s' % closed_loop.__file__)
print('       controller.py   -> %s' % gantry_dynamic.controller.__file__)
print('       plant.py        -> %s (TRAJ %s)' % (plant.__file__, plant.TRAJ))

entry_file = os.path.abspath(entry.__file__)
names = TRAIN_FILES + VAL_FILES + TEST_FILES
td = traj_dir(cfg)
missing = [f for f in names if not os.path.isfile(os.path.join(td, f))]
res['c2_entry_file'] = entry_file
res['c2_entry_is_repo_entry'] = entry_file == os.path.abspath(bbcl.ENTRY)
res['c2_cfg_type_module'] = type(CFG).__module__ + ' @ ' + sys.modules[type(CFG).__module__].__file__
res['c2_traj_dir'] = td
res['c2_n_records'] = [len(TRAIN_FILES), len(VAL_FILES), len(TEST_FILES)]
res['c2_missing'] = missing
print('[G0.2] CFG from %s (repo entry: %s); RunConfig class from %s'
      % (entry_file, res['c2_entry_is_repo_entry'], res['c2_cfg_type_module']))
print('       traj_dir %s: %d/%d/%d records, missing %s'
      % (td, len(TRAIN_FILES), len(VAL_FILES), len(TEST_FILES), missing))
print(bbcl.describe_cfg(cfg, CFG))

rep = bbcl.check_vendor()
res['c3_files'] = len(rep)
res['c3_patched'] = [(f, h) for f, h, u in rep if h]
res['c3_unmarked'] = [(f, u) for f, h, u in rep if u]
print('[G0.3] %d vendored .py files compared to their sources; %d carry patch hunks %s; '
      '%d unmarked differences %s' % (len(rep), len(res['c3_patched']), res['c3_patched'],
                                      len(res['c3_unmarked']), res['c3_unmarked']))

res['cfg'] = {k: getattr(cfg, k) for k in ('mode', 'fs_new', 'nf', 'na_nb', 'stride', 'burn_in',
                                           'batch_size', 'epochs', 'its_per_val', 'seed')}
ok = (not bad) and res['c2_entry_is_repo_entry'] and not missing and not res['c3_unmarked'] \
    and len(names) == 29
res['PASS'] = bool(ok)
os.makedirs(os.path.join(bbcl.OUT, 'g0'), exist_ok=True)
json.dump(res, open(os.path.join(bbcl.OUT, 'g0', 'g0.json'), 'w'), indent=1, default=str)
print('[G0] %s' % ('PASS' if ok else 'FAIL'))
