"""Layers 3b and 3c of design 6.3, one case on one side per process.

    python -u run_l3bc.py --side S --layer 3b --row 221 --noise lowpass
    python -u run_l3bc.py --side S --layer 3c --row 226 --noise lowpass --kind i|ii|iii

3b: the GOLDEN (mainline) layer-3 `_pre_lbfgs` pair is loaded into a freshly built model on each
side and `lbfgs_polish(..., val_before=0.0)` forces the reject path (restore and _resync). Saves the
outcome and the state after it, and asserts in-process that the parameters equal the start.

3c: reruns from each code's OWN layer-3 golden (D-275 D3): (i) Adam continuation from the deepSI
`_best.pth` with epochs = 4, (ii) polish-only from the `_pre_lbfgs` pair, (iii) (i) from `_last.pth`.
Mainline: RESUME_CHECKPOINT = a copy of the golden source and EP.CFG = the budget (start_phase
'lbfgs' for ii) in a new run folder. Clean: main(cfg, rerun=...) on a copy of the clean golden run
folder and of its deepSI dumps; the cfg comes through RunConfig.from_config_json.
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
ap.add_argument('--layer', required=True, choices=('3b', '3c'))
ap.add_argument('--row', type=int, required=True, choices=sorted(H.ROWS))
ap.add_argument('--noise', required=True, choices=('lowpass', 'noisefree'))
ap.add_argument('--kind', choices=('i', 'ii', 'iii'))
args = ap.parse_args()

case = '%d-%s' % (args.row, args.noise)
tag = case if args.layer == '3b' else '%s-%s' % (case, args.kind)
side_base = H.GOLDEN if args.side == 'mainline' else H.CLEAN_OUT
out = os.path.join(side_base, 'L' + args.layer, tag)
if os.path.exists(out):
    sys.exit('%s exists; remove it first' % out)
os.makedirs(out)
lad = os.path.join(out, 'localappdata')
os.environ['LOCALAPPDATA'] = lad
l3_src = os.path.join(side_base, 'L3', case)          # this side's own layer-3 golden (3c)
l3_golden = os.path.join(H.GOLDEN, 'L3', case)          # the mainline golden (3b)
rid_l3 = 'L3-' + case
T0 = time.time()

H.select_side(args.side)
import numpy as np
import torch

if args.layer == '3b':
    os.makedirs(lad)
    cfg, EP = H.make_cfg(args.row, args.noise, run_id='L3b-' + case)
    import gantry_dynamic.data as GDD
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop
    from gantry_dynamic.training import load_checkpoint
    import gantry_dynamic.training as GDT
    from model_augmentation.fit_systems.lbfgs_polish import PolishSpec, lbfgs_polish
    H.assert_roots()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = GDD.load_datasets(cfg)
    norm = GDD.compute_normalization(cfg, data)
    hp = cfg.hp
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(hp, cfg, data, norm)
    fit_sys.simulator = build_closed_loop(
        fit_sys, norm, cfg, train_files=data.train_files, val_files=data.val_files,
        val_data=data.val_ckpt_data, e_train=data.e_train, e_val=data.e_val)
    load_checkpoint(fit_sys, os.path.join(l3_golden, 'run', 'gantry_ckpt_%s_pre_lbfgs' % rid_l3),
                    cfg.joint_estimation, start_phase='lbfgs')
    start = {k: v.clone() for k, v in list(fit_sys.hfn.state_dict().items())
             + [('enc.' + k, v) for k, v in fit_sys.encoder.state_dict().items()]}
    if args.side == 'mainline':            # the spec and meters exactly as mainline run_lbfgs_polish
        spec = PolishSpec(
            n_windows=cfg.lbfgs_windows, max_iter=cfg.lbfgs_max_iter,
            inner_iter=cfg.lbfgs_inner_iter, history_size=cfg.lbfgs_history,
            tol_grad=cfg.lbfgs_tol_grad, tol_change=cfg.lbfgs_tol_change,
            freeze=tuple(cfg.lbfgs_freeze), meter_rel_tol=cfg.lbfgs_meter_rel_tol,
            time_budget_s=cfg.lbfgs_time_budget_s, chunk=cfg.lbfgs_chunk, seed=cfg.seed,
            meters_gate=cfg.lbfgs_meters_gate)
        kw = dict(meters_fn=GDT._gantry_meters)
    else:
        spec = PolishSpec(
            n_windows=cfg.lbfgs_windows, max_iter=cfg.lbfgs_max_iter,
            inner_iter=cfg.lbfgs_inner_iter, history_size=cfg.lbfgs_history,
            tol_grad=cfg.lbfgs_tol_grad, tol_change=cfg.lbfgs_tol_change,
            seed=cfg.seed)
        kw = {}
    res = lbfgs_polish(fit_sys, data.train_data, loss_kwargs={'nf': hp['nf'], 'stride': cfg.stride},
                       spec=spec, val_sys_data=data.val_ckpt_data, val_before=0.0,
                       validation_measure=cfg.validation_measure, verbose=True, **kw)
    after = {k: v.clone() for k, v in list(fit_sys.hfn.state_dict().items())
             + [('enc.' + k, v) for k, v in fit_sys.encoder.state_dict().items()]}
    params_restored = all(torch.equal(start[k], after[k]) for k in start
                          if not k.endswith(('theta_frozen', 'vbar')))
    torch.save({'start': start, 'after': after}, os.path.join(out, 'state.pt'))
    with open(os.path.join(out, 'outcome.json'), 'w') as fh:
        json.dump(dict(outcome=res, params_restored=params_restored), fh, indent=1, default=str)
    files = (data.train_files, data.val_files, data.test_files)
    print('L3B accepted=%s params_restored=%s reason=%s' % (res['accepted'], params_restored, res['reason']))
else:
    kind = args.kind
    budget = dict(H.BUDGET, epochs=4) if kind in ('i', 'iii') else dict(H.BUDGET)
    src_name = {'i': 'SSE_Interconnect_Composed_%s_best.pth' % rid_l3,
                'ii': 'gantry_ckpt_%s_pre_lbfgs' % rid_l3,
                'iii': 'SSE_Interconnect_Composed_%s_last.pth' % rid_l3}[kind]
    run_id = 'L3c-' + tag
    shutil.copytree(os.path.join(l3_src, 'localappdata'), lad)       # deepSI dumps of this side's golden
    if args.side == 'mainline':
        src_dir = os.path.join(out, 'source')
        os.makedirs(src_dir)
        if kind == 'ii':
            for ext in ('.pt', '.npz'):
                shutil.copy2(os.path.join(l3_src, 'run', src_name + ext), src_dir)
            resume = os.path.join(src_dir, src_name)
        else:
            shutil.copy2(os.path.join(lad, 'deepSI', 'checkpoints', src_name), src_dir)
            resume = os.path.join(src_dir, src_name)
        os.environ['RESUME_CHECKPOINT'] = resume
        extra = dict(start_phase='lbfgs') if kind == 'ii' else {}
        cfg, EP = H.make_cfg(args.row, args.noise, run_id=run_id, budget=budget, **extra)
        H.assert_roots()
        EP.save_dir = lambda cfg: out
        EP.evaluate_and_save = lambda *a, **k: None
        EP.state_recovery_diagnostic = lambda *a, **k: None
        EP.compute_gradient_norms = lambda *a, **k: ({}, {})
        EP.main()
        os.rename(os.path.join(out, run_id), os.path.join(out, 'run'))
    else:
        run_dir = os.path.join(out, 'run')
        shutil.copytree(os.path.join(l3_src, 'run'), run_dir)
        os.environ['SLURM_JOB_ID'] = run_id                     # the rerun id
        cfg0, EP = H.make_cfg(args.row, args.noise, run_id=run_id, budget=budget)
        H.assert_roots()
        from gantry_dynamic.config import RunConfig
        with open(os.path.join(run_dir, 'config.json')) as fh:
            meta0 = json.load(fh)
        cfg = RunConfig.from_config_json(meta0, **({'epochs': 4} if kind in ('i', 'iii') else {}))
        assert cfg == cfg0, (cfg, cfg0)
        EP.main(cfg, rerun=('resume', run_dir, 'best' if kind == 'i' else 'last') if kind != 'ii'
                else ('polish-only', run_dir, None))
        # clean-only assertions (6.3 layer 3c)
        prev = os.path.join(run_dir, 'previous', run_id)
        moved = sorted(os.listdir(prev))

        def _orig(f):
            p = os.path.join(l3_src, 'run', f)
            return p if os.path.exists(p) else os.path.join(l3_src, 'localappdata', 'deepSI', 'checkpoints', f)
        expected = sorted(f for f in os.listdir(os.path.join(l3_src, 'run')) if f != 'config.json')
        if kind != 'ii':
            expected += ['SSE_Interconnect_Composed_%s%s.pth' % (rid_l3, s) for s in ('_best', '_last')]
        same = (sorted(moved) == sorted(expected)
                and all(open(os.path.join(prev, f), 'rb').read() == open(_orig(f), 'rb').read()
                        for f in moved))
        meta1 = json.load(open(os.path.join(run_dir, 'config.json')))
        only_resumes = ({k: v for k, v in meta1.items() if k != 'RESUMES'} == meta0
                        and len(meta1.get('RESUMES', [])) == 1)
        with open(os.path.join(out, 'clean_asserts.json'), 'w') as fh:
            json.dump(dict(previous_files=moved, previous_byte_identical=same,
                           config_only_resumes=only_resumes, resumes=meta1.get('RESUMES')), fh, indent=1)
        print('L3C clean asserts: previous byte-identical=%s, config.json only RESUMES=%s, moved=%s'
              % (same, only_resumes, moved))
    import gantry_dynamic.data as GDD
    files = GDD.record_files(cfg)
H.assert_roots()


class _D:
    train_files, val_files, test_files = files


H.write_meta(out, cfg, _D, dict(case=tag, layer=args.layer, seconds=time.time() - T0))
print('L3BC_DONE side=%s layer=%s case=%s seconds=%.1f' % (args.side, args.layer, tag, time.time() - T0),
      flush=True)
