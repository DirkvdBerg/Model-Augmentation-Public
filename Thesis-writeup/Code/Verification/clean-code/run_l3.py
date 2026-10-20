"""Layer 3 of design 6.3 (short seeded training) for one case, on one side, in one process.

    python -u run_l3.py --side mainline|clean --row 221 --noise lowpass

Writes <base>/L3/<row>-<noise>/: the run folder `run/` (config.json, both checkpoint pairs,
val_hist.npz) and deepSI's dumps under `localappdata/deepSI/checkpoints/` (LOCALAPPDATA is
redirected there, 6.2; layer 3c resumes from them). <base> is simulations/clean-golden (mainline,
the golden) or simulations/clean-golden/_clean (clean side).

Mainline: EP.main() with EP.CFG replaced by the harness budget (6.2). Two harness-side
substitutions, neither of which touches a number: EP.save_dir points at the golden folder instead
of the mainline family folder, and the post-run report and diagnostics (EP:456-471, after the final
checkpoint is written, H2) are skipped.
"""
import argparse
import os
import shutil
import sys
import time

import hcommon as H

ap = argparse.ArgumentParser()
ap.add_argument('--side', required=True, choices=('mainline', 'clean'))
ap.add_argument('--row', type=int, required=True, choices=sorted(H.ROWS))
ap.add_argument('--noise', required=True, choices=('lowpass', 'noisefree'))
ap.add_argument('--meta-only', action='store_true', help='write meta.json for a finished run folder')
args = ap.parse_args()

case = '%d-%s' % (args.row, args.noise)
base = os.path.join(H.GOLDEN if args.side == 'mainline' else H.CLEAN_OUT, 'L3', case)
if os.path.exists(base) and not args.meta_only:
    sys.exit('%s exists; remove it first (goldens are never overwritten)' % base)
lad = os.path.join(base, 'localappdata')
if not args.meta_only:
    os.makedirs(lad)
os.environ['LOCALAPPDATA'] = lad
run_id = 'L3-' + case

H.select_side(args.side)
T0 = time.time()
cfg, EP = H.make_cfg(args.row, args.noise, run_id=run_id)
H.assert_roots()
if args.meta_only:
    pass
elif args.side == 'mainline':
    EP.save_dir = lambda cfg: base
    EP.evaluate_and_save = lambda *a, **k: None
    EP.state_recovery_diagnostic = lambda *a, **k: None
    EP.compute_gradient_norms = lambda *a, **k: ({}, {})
    EP.main()
    os.rename(os.path.join(base, run_id), os.path.join(base, 'run'))
else:
    import dataclasses
    cfg = dataclasses.replace(cfg, out_root=os.path.join(base, 'out'))
    EP.main(cfg, run_id=run_id)
    src = os.path.join(base, 'out', '%s_data_lsq' % cfg.mode, run_id)
    shutil.move(src, os.path.join(base, 'run'))
    shutil.rmtree(os.path.join(base, 'out'), ignore_errors=True)   # OneDrive may lock the emptied folder
H.assert_roots()

import gantry_dynamic.data as GDD                    # noqa: E402  (records hashed for meta.json)
files = GDD.record_files(cfg)


class _D:
    train_files, val_files, test_files = files


_secs = time.time() - T0
if args.meta_only:                      # the run's own time, from its log (SEQ_END line)
    _secs = float(os.environ.get('L3_SECONDS', 'nan'))
H.write_meta(base, cfg, _D, dict(case=case, layer='3', run_id=run_id, seconds=_secs))
print('L3_DONE side=%s case=%s seconds=%.1f' % (args.side, case, time.time() - T0), flush=True)
