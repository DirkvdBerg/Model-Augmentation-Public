"""D-295: the PS2 opening and the training after it from a COMMON start, on one side, or the comparison.

    python -u ps2_common_start.py --side mainline|clean [--row 221]
    python -u ps2_common_start.py --compare [--row 221]

Like run_l3.py (layer 3, the harness budget), but both sides start from the same model state: mainline saves its
hfn and encoder state right after its encoder start (prefit_physical_encoder), and the clean side, after its own
encoder start, loads that state (hcommon.drop_state for the renamed and dropped keys). The baseline constants that
D-294 changes (theta^0, P_mat, Lb, C) are buffers of that state, so on tree B (pinv_revert.py) the two sides then
hold the same model and every difference in the PS opening, the OBC refresh, Adam and the polish is the code's.
Mainline writes HARNESS_GOLDEN/PS2/<row>-lowpass/, the clean side HARNESS_CLEAN_OUT/PS2/<row>-lowpass/; --compare
runs compare_l3.compare_runs on both run folders (pairs, Adam state, .npz values with ps2_summary, val_hist, config).
"""
import argparse
import json
import os
import shutil
import sys
import time

import hcommon as H

ap = argparse.ArgumentParser()
ap.add_argument('--side', choices=('mainline', 'clean'))
ap.add_argument('--compare', action='store_true')
ap.add_argument('--row', type=int, default=221, choices=sorted(H.ROWS))
args = ap.parse_args()
case = '%d-lowpass' % args.row
gdir, cdir = os.path.join(H.GOLDEN, 'PS2', case), os.path.join(H.CLEAN_OUT, 'PS2', case)
state_file = os.path.join(gdir, 'common_start.pt')
run_id = 'PS2-' + case

if args.compare:
    from compare_l3 import compare_runs
    out = {}
    compare_runs(os.path.join(gdir, 'run'), os.path.join(cdir, 'run'), out, run_id, run_id)
    for k, v in list(out.items())[:25]:
        print('    %s: %s' % (k, v))
    print('COMPARE_PS2 %s %s (%d mismatches)' % (case, 'FAIL' if out else 'PASS', len(out)), flush=True)
    sys.exit(1 if out else 0)
if args.side is None:
    sys.exit('give --side or --compare')

base = gdir if args.side == 'mainline' else cdir
if os.path.exists(base) and args.side == 'clean' or os.path.exists(os.path.join(base, 'run')):
    sys.exit('%s exists; remove it first' % base)
if args.side == 'clean' and not os.path.exists(state_file):
    sys.exit('no common start at %s: run the mainline side first' % state_file)
os.makedirs(base, exist_ok=True)
os.environ['LOCALAPPDATA'] = os.path.join(base, 'localappdata')
os.makedirs(os.environ['LOCALAPPDATA'], exist_ok=True)
T0 = time.time()
H.select_side(args.side)
import torch  # noqa: E402

cfg, EP = H.make_cfg(args.row, 'lowpass', run_id=run_id)
H.assert_roots()

# the one injection point: right after each side's own encoder start
name = 'prefit_physical_encoder' if args.side == 'mainline' else 'encoder_start'
_start = getattr(EP, name)


def _common_start(fit_sys, *a, **k):
    out = _start(fit_sys, *a, **k)
    if args.side == 'mainline':
        torch.save({'hfn': fit_sys.hfn.state_dict(), 'encoder': fit_sys.encoder.state_dict()}, state_file)
        print('[ps2 common start] mainline state saved: %s' % state_file, flush=True)
    else:
        st = torch.load(state_file, map_location='cpu')
        fit_sys.hfn.load_state_dict(H.drop_state(st['hfn']))
        fit_sys.encoder.load_state_dict(st['encoder'])
        # the OBC basis J was built at model build from the clean constants (D-294); rebuild it at epoch 0 from the
        # loaded ones, as mainline built its own
        if fit_sys.obc is not None:
            with torch.no_grad():
                fit_sys.obc.refresh(fit_sys.hfn.connected_blocks[fit_sys._mats_ix].beta, 0)
        print('[ps2 common start] mainline state loaded into the clean model', flush=True)
    return out


setattr(EP, name, _common_start)
if args.side == 'mainline':
    EP.save_dir = lambda cfg: base
    EP.evaluate_and_save = lambda *a, **k: None
    EP.state_recovery_diagnostic = lambda *a, **k: None
    EP.compute_gradient_norms = lambda *a, **k: ({}, {})
    EP.main()
    os.rename(os.path.join(base, run_id), os.path.join(base, 'run'))
else:
    import dataclasses
    cfg = dataclasses.replace(cfg, out_root=os.path.join(base, 'out'))
    EP.main(cfg, run_id=run_id, evaluate_run=False)
    shutil.move(os.path.join(base, 'out', cfg.noise, run_id), os.path.join(base, 'run'))
    shutil.rmtree(os.path.join(base, 'out'), ignore_errors=True)
H.assert_roots()
import gantry_dynamic.data as GDD  # noqa: E402


class _D:
    train_files, val_files, test_files = GDD.record_files(cfg)


H.write_meta(base, cfg, _D, dict(case=case, layer='PS2 common start', run_id=run_id, seconds=time.time() - T0))
with open(os.path.join(base, 'ps2_summary.json'), 'w') as fh:
    import numpy as np
    with np.load(os.path.join(base, 'run', 'gantry_ckpt_%s.npz' % run_id), allow_pickle=True) as z:
        json.dump(dict(ps2_summary=str(z['ps2_summary']) if 'ps2_summary' in z.files else None), fh, indent=1)
print('PS2_COMMON_DONE side=%s case=%s seconds=%.1f' % (args.side, case, time.time() - T0), flush=True)
