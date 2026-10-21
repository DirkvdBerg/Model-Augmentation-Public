"""Layers 3b and 3c of design 6.3, one case on one side per process.

    python -u run_l3bc.py --side S --layer 3b --row 221 --noise lowpass
    python -u run_l3bc.py --side S --layer 3c --row 226 --noise lowpass --kind i|ii|iii|iv|v|vi|vii

3b: the GOLDEN (mainline) layer-3 `_pre_lbfgs` pair is loaded into a freshly built model on each
side and `lbfgs_polish(..., val_before=0.0)` forces the reject path (restore and _resync). Saves the
outcome and the state after it, and asserts in-process that the parameters equal the start.

3c: reruns from each code's OWN layer-3 golden (D-275 D3): (i) Adam continuation from the deepSI
`_best.pth` with epochs = 4, (ii) polish-only from the `_pre_lbfgs` pair, (iii) (i) from `_last.pth`,
(iv) (iii) with epochs = 3, the spent budget (D-278 F2). Mainline: RESUME_CHECKPOINT = a copy of the
golden source and EP.CFG = the budget (start_phase 'lbfgs' for ii) in a new run folder. Clean:
main(cfg, rerun=...) on a copy of the clean golden run folder and of its deepSI dumps; the cfg comes
through RunConfig.from_config_json. ii and iv also write d278_checks.json on both sides: ii's polish
decision used the source's selected score `bestfit` (D-278 F3), iv ran no Adam update.
Clean only (step 4c, written to clean_asserts.json): (v) a fresh launch into the existing run folder
is refused and the folder is unchanged (F1); (vi) a resume from a truncated `_best.pth`, and on a PS2
row from a dump marked inside an unfinished PS2 opening, is refused before anything moves (F6, F4);
(vii) a second continuation, from case i's `_last.pth` with epochs = 5, resumes at the cumulative
count 4, not at its own 2 (F4).
"""
import argparse
import json
import os
import shutil
import sys
import time

import hcommon as H

CLEAN_ONLY = ('v', 'vi', 'vii')

ap = argparse.ArgumentParser()
ap.add_argument('--side', required=True, choices=('mainline', 'clean'))
ap.add_argument('--layer', required=True, choices=('3b', '3c'))
ap.add_argument('--row', type=int, required=True, choices=sorted(H.ROWS))
ap.add_argument('--noise', required=True, choices=('lowpass', 'noisefree'))
ap.add_argument('--kind', choices=('i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii'))
args = ap.parse_args()
if args.layer == '3c' and args.side == 'mainline' and args.kind in CLEAN_ONLY:
    sys.exit('kind %s is clean-only' % args.kind)

case = '%d-%s' % (args.row, args.noise)
tag = case if args.layer == '3b' else '%s-%s' % (case, args.kind)
side_base = H.GOLDEN if args.side == 'mainline' else H.CLEAN_OUT
final = os.path.join(side_base, 'L' + args.layer, tag)
if os.path.exists(final):
    sys.exit('%s exists; remove it first' % final)
out = final
if args.layer == '3c' and args.side == 'clean':
    # A clean rerun archives its run folder and at once rewrites val_hist.npz under the same name;
    # in the OneDrive-synced tree that write failed with Errno 22 (3 of 11 reruns, 2026-10-21). The
    # case therefore runs in an unsynced folder and is moved to `final` at the end; no number changes.
    import tempfile
    out = os.path.join(tempfile.gettempdir(), 'clean-harness', 'L3c', tag)
    shutil.rmtree(out, ignore_errors=True)
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
        kw = dict(meters_fn=GDT._gantry_meters, verbose=True)
    else:                                  # C3 (D-281): the clean polish has no `verbose`, it always prints
        spec = PolishSpec(
            n_windows=cfg.lbfgs_windows, max_iter=cfg.lbfgs_max_iter,
            inner_iter=cfg.lbfgs_inner_iter, history_size=cfg.lbfgs_history,
            tol_grad=cfg.lbfgs_tol_grad, tol_change=cfg.lbfgs_tol_change,
            seed=cfg.seed)
        kw = {}
    res = lbfgs_polish(fit_sys, data.train_data, loss_kwargs={'nf': hp['nf'], 'stride': cfg.stride},
                       spec=spec, val_sys_data=data.val_ckpt_data, val_before=0.0,
                       validation_measure=cfg.validation_measure, **kw)
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
    epochs = {'i': 4, 'iii': 4, 'vii': 5}.get(kind)          # None: the budget's own 3 (iv: spent)
    budget = dict(H.BUDGET, epochs=epochs) if epochs else dict(H.BUDGET)
    src_name = {'i': 'SSE_Interconnect_Composed_%s_best.pth' % rid_l3,
                'ii': 'gantry_ckpt_%s_pre_lbfgs' % rid_l3,
                'iii': 'SSE_Interconnect_Composed_%s_last.pth' % rid_l3,
                'iv': 'SSE_Interconnect_Composed_%s_last.pth' % rid_l3,
                'vi': 'SSE_Interconnect_Composed_%s_best.pth' % rid_l3,
                'vii': 'SSE_Interconnect_Composed_%s_last.pth' % rid_l3}.get(kind)
    if kind == 'vii':                          # the second continuation starts from case i's output
        l3_src = os.path.join(side_base, 'L3c', case + '-i')
    run_id = 'L3c-' + tag
    shutil.copytree(os.path.join(l3_src, 'localappdata'), lad)       # deepSI dumps of this side's source
    ckdir = os.path.join(lad, 'deepSI', 'checkpoints')
    run_dir = os.path.join(out, 'run')

    def _tree(d):
        """{relative path: SHA256} of every file below d."""
        return {os.path.relpath(os.path.join(r, f), d): H.sha256_file(os.path.join(r, f))
                for r, _, fs in os.walk(d) for f in fs}
    if args.side == 'mainline':
        src_dir = os.path.join(out, 'source')
        os.makedirs(src_dir)
        if kind == 'ii':
            for ext in ('.pt', '.npz'):
                shutil.copy2(os.path.join(l3_src, 'run', src_name + ext), src_dir)
        else:
            shutil.copy2(os.path.join(ckdir, src_name), src_dir)
        os.environ['RESUME_CHECKPOINT'] = os.path.join(src_dir, src_name)
        extra = dict(start_phase='lbfgs') if kind == 'ii' else {}
        cfg, EP = H.make_cfg(args.row, args.noise, run_id=run_id, budget=budget, **extra)
        H.assert_roots()
        EP.save_dir = lambda cfg: out
        EP.evaluate_and_save = lambda *a, **k: None
        EP.state_recovery_diagnostic = lambda *a, **k: None
        EP.compute_gradient_norms = lambda *a, **k: ({}, {})
        EP.main()
        os.rename(os.path.join(out, run_id), run_dir)
    else:
        shutil.copytree(os.path.join(l3_src, 'run'), run_dir)
        os.environ['SLURM_JOB_ID'] = run_id                     # the rerun id
        cfg0, EP = H.make_cfg(args.row, args.noise, run_id=run_id, budget=budget)
        H.assert_roots()
        from gantry_dynamic.config import RunConfig, save_dir
        with open(os.path.join(run_dir, 'config.json')) as fh:
            meta0 = json.load(fh)
        cfg = RunConfig.from_config_json(meta0, **({'epochs': epochs} if epochs else {}))
        assert cfg == cfg0, (cfg, cfg0)
        if kind == 'v':
            # F1: the layer-3 run folder placed where a fresh launch with its run id would write
            from dataclasses import replace
            cfg = replace(cfg, out_root=os.path.join(out, 'out'))
            sdir = os.path.join(save_dir(cfg), rid_l3)
            shutil.copytree(run_dir, sdir)
            before = _tree(sdir)
            try:
                EP.main(cfg, run_id=rid_l3)
                r = dict(refused=False)
            except FileExistsError as e:
                r = dict(refused=True, error=str(e))
            r['unchanged'] = _tree(sdir) == before
            asserts = dict(r, ok=bool(r['refused'] and r['unchanged']))
        elif kind == 'vi':
            # F6 (and F4 on a PS2 row): refused before anything moves
            best = os.path.join(ckdir, src_name)
            with open(best, 'rb') as fh:
                good = fh.read()
            attempts = {}

            def _attempt(label, need):
                before = (_tree(run_dir), _tree(ckdir))
                try:
                    EP.main(cfg, rerun=('resume', run_dir, 'best'))
                    r = dict(refused=False)
                except Exception as e:
                    r = dict(refused=True, error='%s: %s' % (type(e).__name__, str(e).splitlines()[0][:300]))
                r['unchanged'] = ((_tree(run_dir), _tree(ckdir)) == before
                                  and not os.path.exists(os.path.join(run_dir, 'previous')))
                r['ok'] = bool(r['refused'] and r['unchanged'] and need in r.get('error', ''))
                attempts[label] = r
                print('L3C vi %s: %s' % (label, r), flush=True)
            with open(best, 'wb') as fh:
                fh.write(good[:len(good) // 2])
            _attempt('truncated _best.pth', '')
            if H.ROWS[args.row][4]:                              # a PS2 row
                with open(best, 'wb') as fh:
                    fh.write(good)
                d = torch.load(best, map_location='cpu', weights_only=False)
                d['ps2_opening'] = ('S1', 3)                    # 3 updates into S1
                torch.save(d, best)
                _attempt('dump inside an unfinished PS2 opening', 'PS2')
            with open(best, 'wb') as fh:
                fh.write(good)
            asserts = dict(attempts=attempts, ok=bool(
                len(attempts) == (2 if H.ROWS[args.row][4] else 1) and all(a['ok'] for a in attempts.values())))
        else:
            EP.main(cfg, rerun=('resume', run_dir, 'best' if kind == 'i' else 'last') if kind != 'ii'
                    else ('polish-only', run_dir, None))
            # clean-only assertions (6.3 layer 3c)
            prev = os.path.join(run_dir, 'previous', run_id)
            moved = sorted(os.listdir(prev))

            def _orig(f):
                p = os.path.join(l3_src, 'run', f)
                return p if os.path.exists(p) else os.path.join(l3_src, 'localappdata', 'deepSI', 'checkpoints', f)
            expected = sorted(f for f in os.listdir(os.path.join(l3_src, 'run')) if f not in ('config.json', 'previous'))
            if kind != 'ii':
                expected += ['SSE_Interconnect_Composed_%s%s.pth' % (rid_l3, s) for s in ('_best', '_last')]
            same = (sorted(moved) == sorted(expected)
                    and all(open(os.path.join(prev, f), 'rb').read() == open(_orig(f), 'rb').read()
                            for f in moved))
            meta1 = json.load(open(os.path.join(run_dir, 'config.json')))
            strip = lambda m: {k: v for k, v in m.items() if k != 'RESUMES'}             # noqa: E731
            only_resumes = (strip(meta1) == strip(meta0)
                            and len(meta1.get('RESUMES', [])) == len(meta0.get('RESUMES', [])) + 1)
            asserts = dict(previous_files=moved, previous_byte_identical=same,
                           config_only_resumes=only_resumes, resumes=meta1.get('RESUMES'))
            ok = same and only_resumes
            if kind == 'vii':
                d = torch.load(os.path.join(ckdir, 'SSE_Interconnect_Composed_%s_last.pth' % rid_l3),
                               map_location='cpu', weights_only=False)
                # closed after the read: an open handle blocks the move out of the temp folder (WinError 5)
                with np.load(os.path.join(run_dir, 'gantry_ckpt_%s.npz' % rid_l3), allow_pickle=True) as z:
                    chain = dict(resumed_at_epoch=int(z['resumed_at_epoch']), done_epochs=int(z['done_epochs']),
                                 last_epochs_before=d.get('epochs_before'),
                                 last_epoch_counter=float(d.get('epoch_counter')))
                chain['ok'] = bool(chain['resumed_at_epoch'] == 4 and chain['done_epochs'] == 5
                                   and chain['last_epochs_before'] == 4
                                   and round(chain['last_epoch_counter'], 9) == 1.0)
                asserts['chain'] = chain
                ok = ok and chain['ok']
            asserts['ok'] = bool(ok)
        with open(os.path.join(out, 'clean_asserts.json'), 'w') as fh:
            json.dump(asserts, fh, indent=1)
        print('L3C clean asserts: %s' % json.dumps(asserts), flush=True)
    if kind in ('ii', 'iv'):
        # D-278, both sides: ii's polish was decided against the source's selected score, iv ran no Adam
        # every np.load below is closed after its read: an open handle blocks the move out of the
        # temp folder (WinError 5)
        with np.load(os.path.join(run_dir, 'gantry_ckpt_%s.npz' % (run_id if args.side == 'mainline' else rid_l3)),
                     allow_pickle=True) as z:
            oc = json.loads(str(z['lbfgs_outcome']))
            if kind == 'iv':
                resumed_at, done = int(z['resumed_at_epoch']), int(z['done_epochs'])
        if kind == 'ii':
            src_npz = (os.path.join(out, 'source', src_name + '.npz') if args.side == 'mainline'
                       else os.path.join(run_dir, 'previous', run_id, src_name + '.npz'))
            with np.load(src_npz, allow_pickle=True) as zs:
                sel = float(zs['bestfit'])
            chk = dict(val_before=oc['val_before'], selected_bestfit=sel, accepted=oc['accepted'],
                       ok=bool(np.isfinite(sel) and oc['val_before'] == sel))
        else:
            vh = os.path.join(run_dir, 'val_hist.npz')
            ev = []
            if os.path.exists(vh):
                with np.load(vh) as zv:
                    ev = [str(e) for e in zv['val_hist_event']]
            chk = dict(resumed_at_epoch=resumed_at, done_epochs=done,
                       events=ev, polish_ran=None if oc is None else oc['ran'],
                       ok=bool(resumed_at == 3 and 'adam' not in ev and oc is not None))
        with open(os.path.join(out, 'd278_checks.json'), 'w') as fh:
            json.dump(chk, fh, indent=1)
        print('L3C D-278 checks: %s' % json.dumps(chk), flush=True)
    import gantry_dynamic.data as GDD
    files = GDD.record_files(cfg)
H.assert_roots()


class _D:
    train_files, val_files, test_files = files


if out != final:
    os.makedirs(os.path.dirname(final), exist_ok=True)     # the first case into a fresh _clean/L3c/
    shutil.move(out, final)
    out = final
H.write_meta(out, cfg, _D, dict(case=tag, layer=args.layer, seconds=time.time() - T0))
print('L3BC_DONE side=%s layer=%s case=%s seconds=%.1f' % (args.side, args.layer, tag, time.time() - T0),
      flush=True)
