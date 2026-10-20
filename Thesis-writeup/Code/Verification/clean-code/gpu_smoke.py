"""The compiled GPU smokes of design 6.4, one code root per process (server only).

    (a) python -u gpu_smoke.py --side S --row 221|226 --out DIR
        the smoke budget WITH compilation: truncate_s 2.0, n_its 4, its_per_val 4, lbfgs_inner_iter 1,
        lbfgs_max_iter 2, lbfgs_windows 256, ps2_smoke, device cuda, compile_mode reduce-overhead.
        Asserted in this process: the compiled branch RAN (torch._dynamo unique_graphs > 0 after
        fit) and both checkpoint pairs exist. Recorded for the cross-code comparison: the initial
        validation score and the training losses of the four Adam updates.
    (b) python -u gpu_smoke.py --side S --row 226 --polish-from RUN_DIR --out DIR
        polish-only from (a)'s `_pre_lbfgs` pair of the same side: fit() never runs, so the polish
        takes the compiled try with its eager fallback and second probe. Recorded: deterministic,
        used_compiled, loss_first.
Mainline gets THESIS_SMOKE=1 only for the PS2 smoke spec (training.py:875); its EP.CFG is replaced by
the budget below, not by THESIS_SMOKE's own replace (which turns compilation off).
Writes DIR/result.json; compare the two sides with compare_gpu.py.
"""
import argparse
import json
import os
import shutil
import sys
import time

import hcommon as H

ap = argparse.ArgumentParser()
ap.add_argument('--side', required=True, choices=('mainline', 'clean'))
ap.add_argument('--row', type=int, required=True, choices=(221, 226))
ap.add_argument('--out', required=True)
ap.add_argument('--polish-from', default=None, help='(b): the run folder of this side\'s (a) smoke')
args = ap.parse_args()
OUT = os.path.abspath(args.out)
if os.path.exists(OUT):
    sys.exit('%s exists' % OUT)
os.makedirs(OUT)
os.environ['LOCALAPPDATA'] = os.path.join(OUT, 'localappdata')
os.makedirs(os.environ['LOCALAPPDATA'])
BUDGET = dict(truncate_s=2.0, n_its=4, its_per_val=4, lbfgs_inner_iter=1, lbfgs_max_iter=2,
              lbfgs_windows=256, device='cuda', compile_mode='reduce-overhead')
case = '%d-lowpass' % args.row
run_id = ('gpuA-' if args.polish_from is None else 'gpuB-') + case

H.select_side(args.side)
import torch

# harness-side recorder of the objective values (the class method is wrapped in THIS process only)
_losses = []
cfg, EP = H.make_cfg(args.row, 'lowpass', run_id=run_id, budget=BUDGET,
                     **({'start_phase': 'lbfgs'} if args.polish_from and args.side == 'mainline' else {}))
from model_augmentation.fit_systems.interconnect import SSE_Interconnect_Composed
_loss = SSE_Interconnect_Composed.loss


def _rec(self, *a, **k):
    L = _loss(self, *a, **k)
    if torch.is_grad_enabled():
        _losses.append(float(L.detach()))
    return L


SSE_Interconnect_Composed.loss = _rec
H.assert_roots()
T0 = time.time()
res = dict(side=args.side, row=args.row, kind='b' if args.polish_from else 'a', run_id=run_id,
           cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if args.polish_from is None:
    if args.side == 'mainline':
        EP.save_dir = lambda cfg: OUT
        EP.evaluate_and_save = lambda *a, **k: None
        EP.state_recovery_diagnostic = lambda *a, **k: None
        EP.compute_gradient_norms = lambda *a, **k: ({}, {})
        EP.main()
        run_dir = os.path.join(OUT, run_id)
    else:
        import dataclasses
        cfg = dataclasses.replace(cfg, out_root=OUT)
        EP.main(cfg, run_id=run_id)
        run_dir = os.path.join(OUT, '%s_data_lsq' % cfg.mode, run_id)
    from torch._dynamo.utils import counters
    res['unique_graphs'] = int(counters['stats']['unique_graphs'])
    res['adam_losses'] = _losses[:4]
    res['pairs_written'] = all(os.path.exists(os.path.join(run_dir, 'gantry_ckpt_%s%s%s' % (run_id, s, e)))
                               for s in ('', '_pre_lbfgs') for e in ('.pt', '.npz'))
    import numpy as np
    vh = np.load(os.path.join(run_dir, 'val_hist.npz'))
    res['initial_validation'] = float(vh['val_hist_score'][0])
    res['ok'] = bool(res['unique_graphs'] > 0 and res['pairs_written'] and len(res['adam_losses']) == 4)
else:
    src = os.path.abspath(args.polish_from)
    src_rid = [f for f in os.listdir(src) if f.endswith('_pre_lbfgs.pt')][0][len('gantry_ckpt_'):-len('_pre_lbfgs.pt')]
    if args.side == 'mainline':
        os.makedirs(os.path.join(OUT, 'source'))
        for e in ('.pt', '.npz'):
            shutil.copy2(os.path.join(src, 'gantry_ckpt_%s_pre_lbfgs%s' % (src_rid, e)), os.path.join(OUT, 'source'))
        os.environ['RESUME_CHECKPOINT'] = os.path.join(OUT, 'source', 'gantry_ckpt_%s_pre_lbfgs' % src_rid)
        EP.save_dir = lambda cfg: OUT
        EP.evaluate_and_save = lambda *a, **k: None
        EP.state_recovery_diagnostic = lambda *a, **k: None
        EP.compute_gradient_norms = lambda *a, **k: ({}, {})
        EP.main()
        run_dir, rid = os.path.join(OUT, run_id), run_id
    else:
        from gantry_dynamic.config import RunConfig
        run_dir = os.path.join(OUT, 'run')
        shutil.copytree(src, run_dir)
        os.environ['SLURM_JOB_ID'] = run_id
        cfg = RunConfig.from_config_json(json.load(open(os.path.join(run_dir, 'config.json'))))
        EP.main(cfg, rerun=('polish-only', run_dir, None))
        rid = src_rid
    import numpy as np
    out = json.loads(str(np.load(os.path.join(run_dir, 'gantry_ckpt_%s.npz' % rid), allow_pickle=True)['lbfgs_outcome']))
    res.update(deterministic=out['deterministic'], used_compiled=out['used_compiled'],
               loss_first=out['loss_first'], accepted=out['accepted'], reason=out['reason'])
    res['ok'] = True
H.assert_roots()
res['seconds'] = time.time() - T0
with open(os.path.join(OUT, 'result.json'), 'w') as fh:
    json.dump(res, fh, indent=1)
print('GPU_SMOKE_DONE %s' % json.dumps(res), flush=True)
sys.exit(0 if res['ok'] else 1)
