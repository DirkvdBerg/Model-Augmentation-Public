"""Layers 4 and 4b of design 6.3 (as amended by D-275 amendment 4a): a harness-local loader.

    python -u l4_loader.py --side mainline|clean --staging DIR --out DIR [--runs R1,R4] [--parts l4,l4b]
                           [--eval-check]

For every staged final run (the folder of 6.5, one <job>/ per row) it builds the model from the
run's config.json (clean: RunConfig.from_config_json; mainline: eval_greybox.run_config, the
mainline construction), loads the component pair `gantry_ckpt_<job>` and computes:
  l4   per split (train, val, test) and record, from the model's own k0: y, sigma_e and the
       closed-loop prediction of `model` (all runs) and of `physics`, `base_start`, `base_true`
       (R1, R4, R6, R13); RMS and NRMS_e per record and axis; params_learned; on val the
       selection measure val_score against the recorded `bestfit` (ref_score), asserted finite,
       positive and within 1e-3 relative (the existing CPU-eager vs GPU-compiled acceptance).
  l4b  overlap and bias rows (R1, R4, R13), the deployed basis of R1 on the training stack,
       negation rows (R1, R4), added-state poles (R4, R7) and the frozen-Y baseline of R4 on val.
Every computation is the same harness code on both roots; only the code root differs. With
--eval-check (mainline side) the loader is first proven against the unmodified mainline evaluation:
eval_greybox.evaluate (its cache redirected into --out, never into Thesis-writeup/Results),
eval_overlap.evaluate, eval_negation.evaluate and inspect_checkpoint.added_poles, compared bitwise.
summary.json records the checks and the clean source manifest (H.clean_tree_hashes, H2).
"""
import argparse
import json
import os
import sys
import time

import hcommon as H

ap = argparse.ArgumentParser()
ap.add_argument('--side', required=True, choices=('mainline', 'clean'))
ap.add_argument('--staging', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--runs', default='R1,R2,R3,R4,R5,R6,R7,R13,R14')
ap.add_argument('--parts', default='l4,l4b')
ap.add_argument('--eval-check', action='store_true')
ap.add_argument('--truncate-s', type=float, default=None,
                help='LOCAL SCRIPT TEST ONLY (short records); never for the layer-4 result')
args = ap.parse_args()
STAGING = os.path.abspath(args.staging)
OUT = os.path.abspath(args.out)
os.makedirs(OUT, exist_ok=True)
if args.eval_check and args.side != 'mainline':
    sys.exit('--eval-check runs on the mainline side only')

H.select_side(args.side)
import numpy as np
import torch

# The default CPU least-squares driver is pinned to gelsd in this process (both sides, and the
# unmodified evaluation that --eval-check calls): torch's default gelsy is not repeatable from call to
# call at one thread (local test 2026-10-20: up to 3e-13 relative on the same tensors, also under
# MKL_CBWR), gelsd is. The training path already passes driver='gelsd' (gantry_dynamic/model.py).
LSTSQ_DRIVER = 'gelsd'
_lstsq = torch.linalg.lstsq


def _lstsq_pinned(A, B, rcond=None, *, driver=None):
    return _lstsq(A, B, rcond=rcond, driver=driver or (LSTSQ_DRIVER if A.device.type == 'cpu' else None))


torch.linalg.lstsq = _lstsq_pinned

FINAL_DESIGN = {'R1': (True, 2, True, True), 'R2': (True, 2, False, True), 'R3': (False, 2, False, False),
                'R4': (False, 2, True, True), 'R5': (False, 2, False, True), 'R6': (False, 0, False, False),
                'R7': (False, 8, False, True), 'R13': (True, 0, True, False), 'R14': (False, 0, True, False)}
BASE_ROWS = ('R1', 'R4', 'R6', 'R13')
INTERP = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']
STACKS = ('train', 'val', 'test')


def discover(staging):
    """R label -> job id of the staged runs (eval_common.discover_runs, harness-local)."""
    found = {}
    for d in sorted(os.listdir(staging)):
        p = os.path.join(staging, d, 'config.json')
        if not os.path.exists(p):
            continue
        c = json.load(open(p, encoding='utf-8'))
        key = (bool(c['OBC']), int(c['NX_ANN']), c.get('COMBO_INIT_DETUNE') is not None, bool(c.get('PS2')))
        for run, design in FINAL_DESIGN.items():
            if design == key:
                if run in found:
                    raise ValueError('two staged runs of %s' % run)
                found[run] = int(c['run_id'])
    return found


def record_name(f):
    return os.path.basename(str(f).replace('\\', '/')).split('_')[0]


def run_cfg(run_dir):
    from dataclasses import replace
    meta = json.load(open(os.path.join(run_dir, 'config.json'), encoding='utf-8'))
    if args.side == 'clean':
        from gantry_dynamic.config import RunConfig
        cfg = RunConfig.from_config_json(meta, device='cpu', compile_mode=None, truncate_s=None)
    else:
        from eval_greybox import run_config
        from pathlib import Path
        cfg = run_config(Path(run_dir))
    return cfg if args.truncate_s is None else replace(cfg, truncate_s=args.truncate_s)


def build(cfg):
    """eval_greybox.build, harness-local (same statements)."""
    from gantry_dynamic.data import load_datasets, compute_normalization
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data = load_datasets(cfg); norm = compute_normalization(cfg, data)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    fs = build_model(cfg.hp, cfg, data, norm)
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=data.train_files, val_files=data.val_files,
                                     val_data=data.val_ckpt_data, verbose=False,
                                     e_train=data.e_train, e_val=data.e_val)
    return data, norm, fs


def frozen_y(m, y_op=0.0):
    """eval_greybox.frozen_y, harness-local (same statements)."""
    phy = next(b for b in m.hfn.connected_blocks if hasattr(b, 'free_params'))
    with torch.no_grad():
        K, C, A, mh, alpha, beta, gamma_, N0, N1, N2 = phy.pass_mats()
        assert y_op == 0.0
        phy.N_op = N0.detach().clone()
        phy.d_op = (mh * (alpha * gamma_ - beta ** 2)).detach().clone()
        phy.Y_op = float(y_op)
    return m


def added_poles(fs, data, n_xb, na, n_pts=24):
    """inspect_checkpoint.added_poles, harness-local (same statements)."""
    FS_HZ = 4000.0
    rng = np.random.default_rng(11)
    UH, YH, U0 = [], [], []
    for j in range(n_pts):
        sd = fs.norm.transform(data.val_list[j % len(data.val_list)])
        u, y = np.asarray(sd.u), np.asarray(sd.y)
        na_r, nb_r = getattr(fs, 'na_right', 0), getattr(fs, 'nb_right', 0)
        k = int(rng.integers(max(fs.na, fs.nb), len(u) - 2))
        UH.append(u[k - fs.nb:k + nb_r]); YH.append(y[k - fs.na:k + na_r]); U0.append(u[k])
    dt = next(fs.hfn.parameters()).dtype
    T = lambda a: torch.as_tensor(np.ascontiguousarray(np.stack(a)), dtype=dt)              # noqa: E731
    with torch.no_grad():
        x = fs.encoder(T(UH), T(YH)).reshape(len(UH), -1).contiguous()
    u0 = T(U0).contiguous()
    f = lambda xx: fs.hfn(xx, u0)[1].reshape(len(xx), -1).sum(0)                           # noqa: E731
    J = torch.autograd.functional.jacobian(f, x.clone().contiguous()).permute(1, 0, 2).detach().numpy()
    E = [np.linalg.eigvals(j[n_xb:n_xb + na, n_xb:n_xb + na]) for j in J]
    dom = np.median([np.abs(e).max() for e in E])
    pairs = []
    for e in E:
        e = e[e.imag > 1e-9]
        if len(e):
            s = np.log(e) * FS_HZ
            fr, z = np.abs(s.imag) / (2 * np.pi), -s.real / np.abs(s)
            i = int(np.argmin(z)); pairs.append((fr[i], z[i]))
    pairs = np.array(pairs) if pairs else np.full((1, 2), np.nan)
    return dom, len([1 for e in E if np.any(np.abs(e.imag) > 1e-9)]) / len(E), np.nanmedian(pairs[:, 0]), np.nanmedian(pairs[:, 1])


def lsq_overlap(J, v):
    c = torch.linalg.lstsq(J, v.unsqueeze(1)).solution
    return float(torch.linalg.norm(J @ c) / torch.linalg.norm(v))


def deployed_basis(blk, ref, row_ix, cfg, corr, v_hat):
    """eval_overlap.deployed_basis, harness-local (same statements)."""
    from gantry_dynamic.obc_gantry import make_basis_builder
    space = getattr(corr, 'space', 'tangent')
    B, _ = make_basis_builder(blk, ref, row_ix, cfg.obc_ref_chunk, device=torch.device('cpu'),
                              space=space)(v_hat)
    return B


def l4(run, rid, cfg, data, norm, fs, ref_score):
    from gantry_dynamic.controller import build_controller_bank, y_op_for
    from gantry_dynamic.closed_loop_report import _without_learned, _true_free
    from gantry_dynamic.data import servo_sigma
    from model_augmentation.fit_systems.closed_loop import closed_loop_free_run_rms_batch
    phy = next(b for b in fs.hfn.connected_blocks if hasattr(b, 'free_params'))
    with torch.no_grad():
        p_names, p_init, p_learned, _ = phy.estimation_parameters()
    checks = {}
    for split in STACKS:
        t0 = time.time()
        files, sdl, es = [], [], []
        for f, sd, e in zip(getattr(data, split + '_files'), getattr(data, split + '_list'),
                            getattr(data, 'e_' + split)):
            try:
                y_op_for(f, cfg.controller_gain)
            except KeyError:
                continue
            files.append(f); sdl.append(sd); es.append(e)
        bank, rows, _ = build_controller_bank(files, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u,
                                              dtype=cfg.dtype_pt, use_gain=cfg.controller_gain)
        k0 = max(fs.na, fs.nb)
        systems = {'model': fs}
        if run in BASE_ROWS:
            systems['physics'] = _without_learned(fs)
            systems['base_start'] = _without_learned(fs, torch.zeros_like(phy.free_params))
            systems['base_true'] = _without_learned(fs, _true_free(phy))
        out = {'records': np.array([record_name(f) for f in files]), 'k0': k0,
               'y': np.stack([np.asarray(sd.y, float)[k0:] for sd in sdl]),
               'sigma_e': servo_sigma(es, k0),
               'params_learned': np.asarray(p_learned.cpu(), float),
               'params_init': np.asarray(p_init.cpu(), float)}
        for tag, s in systems.items():
            with torch.no_grad():
                _, _, preds = closed_loop_free_run_rms_batch(s, sdl, bank, rows, return_pred=True)
            out[tag] = np.stack([np.asarray(p, float) for p in preds])
            out['rms_' + tag] = np.sqrt(np.mean((out[tag] - out['y']) ** 2, axis=1))
            out['nrmse_' + tag] = out['rms_' + tag] / out['sigma_e']
        if split == 'val':
            val_score = float(np.mean(out['nrmse_model']))
            out['val_score'], out['ref_score'] = val_score, ref_score
            rel = abs(val_score - ref_score) / ref_score
            checks['val_score'] = dict(val_score=val_score, ref_score=ref_score, rel=rel,
                                       ok=bool(np.isfinite(ref_score) and ref_score > 0
                                               and np.isfinite(val_score) and rel < 1e-3))
        np.savez_compressed(os.path.join(OUT, 'l4_%s_%s.npz' % (run, split)), **out)
        print('[%s %d] l4 %s: %d records, %s, %.0f s' % (run, rid, split, len(files), sorted(systems),
                                                         time.time() - t0), flush=True)
    return checks


def l4b(run, rid, cfg, data, norm, fs):
    from gantry_dynamic.closed_loop_report import _true_free, _without_learned
    from gantry_dynamic.obc_gantry import build_reference_set, make_float64_block
    from gantry_dynamic.controller import build_controller_bank
    from model_augmentation.fit_systems.obc import build_stacked_sensitivity
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    from model_augmentation.fit_systems.closed_loop import closed_loop_free_run_rms_batch
    res = {}
    blocks = list(fs.hfn.connected_blocks)
    phy = next(b for b in blocks if hasattr(b, 'free_params'))
    ann = next(b for b in blocks if isinstance(b, Static_ANN_Block))
    route = [int(i) for i in cfg.ann_route_ix]
    row_ix = [i for i in route if i < cfg.nx_phys]
    cols = [route.index(i) for i in row_ix]
    blk = make_float64_block(phy)
    step = lambda v, z: blk.transition_from_free(v, z)               # noqa: E731
    corr = getattr(fs.hfn, 'obc_correction', None)
    v_hat = (corr.vbar if corr is not None else phy.free_params).detach().double()
    files = {'train': data.train_files, 'val': data.val_files,
             'test': [f for f in data.test_files if record_name(f) in INTERP]}
    arrays = {}
    if run in ('R1', 'R4', 'R13'):                       # eval_overlap.evaluate, harness-local
        v_true = _true_free(phy).detach().double()
        c_true = phy.combinations_from_free(v_true).detach().double()
        Jc = torch.autograd.functional.jacobian(lambda v: phy.combinations_from_free(v).double(), v_true)
        scale = c_true.abs().clone()
        names = list(phy.PARAM_NAMES)
        raw = phy.params_init.double()
        scale[list(phy.COMBO_NAMES).index('m_diff')] = 0.5 * (raw[names.index('m1')] + raw[names.index('m2')])
        observed = (phy.combinations_from_free(phy.free_params.detach().double()).detach() - c_true) / scale
        over, bias = [], []
        for stack in STACKS:
            ref = build_reference_set(cfg, norm, nx_ann=cfg.nx_ann, files=files[stack], verbose=False)
            J = build_stacked_sensitivity(step, v_hat, ref.Z_step, row_ix)
            with torch.no_grad():
                F = ann(ref.Z_field.to(next(ann.parameters()).dtype))[:, cols, 0].double().reshape(-1)
                if corr is not None:
                    B = deployed_basis(blk, ref, row_ix, cfg, corr, v_hat)
                    if run == 'R1' and stack == 'train':
                        arrays['deployed_basis_R1_train'] = B.numpy()
                    F = F - B @ corr.theta_frozen.double()
            net = lsq_overlap(J, F)
            Js = build_stacked_sensitivity(step, v_true, ref.Z_step, row_ix)
            with torch.no_grad():
                f_true = blk.transition_from_free(v_true, ref.Z_step.double())[:, :, 0]
            x_mean = torch.as_tensor(np.ravel(norm.x_mean), dtype=torch.float64)
            std_x = torch.as_tensor(np.ravel(norm.std_x), dtype=torch.float64)
            xi = ((ref.x_phys_next_rec - x_mean) / std_x - f_true)[:, row_ix].reshape(-1)
            Phi = Js
            tru = lsq_overlap(Phi, xi)
            dbeta = torch.linalg.lstsq(Phi, xi.unsqueeze(1)).solution.squeeze(1)
            predicted = (Jc @ dbeta) / scale
            over.append(dict(run=run, stack=stack, n_tuples=ref.n, network_overlap=net, truth_overlap=tru))
            for n, p_, o in zip(phy.COMBO_NAMES, predicted.tolist(), observed.tolist()):
                bias.append((run, stack, n, p_, o))
        res['overlap'], res['bias'] = over, bias
    if run in ('R1', 'R4'):                              # eval_negation.evaluate, harness-local
        def cos(a, b):
            return float(a @ b / (torch.linalg.norm(a) * torch.linalg.norm(b)))
        neg = []
        for stack in STACKS:
            ref = build_reference_set(cfg, norm, nx_ann=cfg.nx_ann, files=files[stack], verbose=False)
            J = build_stacked_sensitivity(step, v_hat, ref.Z_step, row_ix)
            with torch.no_grad():
                F = ann(ref.Z_field.to(next(ann.parameters()).dtype))[:, cols, 0].double().reshape(-1)
                if corr is not None:
                    F = F - deployed_basis(blk, ref, row_ix, cfg, corr, v_hat) @ corr.theta_frozen.double()
                Fb = step(v_hat, ref.Z_step.double())[:, row_ix, 0].reshape(-1)
                Fi = Fb - ref.Z_step[:, :cfg.nx_phys, 0].double()[:, row_ix].reshape(-1)
            c = torch.linalg.lstsq(J, Fb.unsqueeze(1)).solution
            Fp = Fb - (J @ c).squeeze(1)
            rec = ref.record_ix.repeat_interleave(len(row_ix))
            for r, nm in enumerate([record_name(n) for n in ref.record_names]):
                m = rec == r
                neg.append((run, stack, nm, cos(F[m], Fi[m]), cos(F[m], Fb[m]), cos(F[m], Fp[m]),
                            float(torch.linalg.norm(F[m]) / torch.linalg.norm(Fb[m]))))
            neg.append((run, stack, 'pooled', cos(F, Fi), cos(F, Fb), cos(F, Fp),
                        float(torch.linalg.norm(F) / torch.linalg.norm(Fb))))
        res['negation'] = neg
    if run in ('R4', 'R7'):
        res['added_poles'] = [float(v) for v in added_poles(fs, data, cfg.nx_phys, cfg.nx_ann)]
    if run == 'R4':                                      # the --lti path, on the val records
        files_v, sdl_v = list(data.val_files), list(data.val_list)
        bank, rows, _ = build_controller_bank(files_v, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u,
                                              dtype=cfg.dtype_pt, use_gain=cfg.controller_gain)
        m = frozen_y(_without_learned(fs, _true_free(phy)))
        with torch.no_grad():
            _, _, preds = closed_loop_free_run_rms_batch(m, sdl_v, bank, rows, return_pred=True)
        arrays['frozen_y_R4_val'] = np.stack([np.asarray(p, float) for p in preds])
    np.savez_compressed(os.path.join(OUT, 'l4b_%s.npz' % run), **arrays)
    with open(os.path.join(OUT, 'l4b_%s.json' % run), 'w') as fh:
        json.dump(res, fh, indent=1)
    return res


def eval_check(run, rid, loader_l4b, parts):
    """The loader against the unmodified mainline evaluation (mainline side only)."""
    from pathlib import Path
    import eval_greybox
    eval_greybox.CACHE = Path(OUT) / 'eval-cache'       # never Thesis-writeup/Results
    rep = {}
    if 'l4' in parts:
        only = ('physics', 'base_start', 'base_true') if run in BASE_ROWS else ()
        for split in STACKS:
            eval_greybox.evaluate(run, only=only, split=split)
            ev = np.load(eval_greybox.cache_path(run, split), allow_pickle=True)
            mine = np.load(os.path.join(OUT, 'l4_%s_%s.npz' % (run, split)), allow_pickle=True)
            for k in ('y', 'sigma_e', 'model', 'physics', 'base_start', 'base_true', 'params_learned',
                      'val_score', 'ref_score'):
                if k in mine.files and k in ev.files:
                    a, b = np.asarray(mine[k]), np.asarray(ev[k])
                    rep['%s/%s' % (split, k)] = bool(a.dtype == b.dtype and a.shape == b.shape
                                                     and a.tobytes() == b.tobytes())
    if 'l4b' in parts:
        if 'overlap' in loader_l4b:
            import eval_overlap
            over, bias = eval_overlap.evaluate(run)
            rep['overlap'] = [(o['network_overlap'], o['truth_overlap']) for o in over] == \
                [(o['network_overlap'], o['truth_overlap']) for o in loader_l4b['overlap']]
            rep['bias'] = [b[3:6] for b in bias] == [b[2:5] for b in loader_l4b['bias']]
        if 'negation' in loader_l4b:
            import eval_negation
            out = eval_negation.evaluate(run)
            ev_rows = [r[4:9] for st in STACKS for r in out[st]]
            rep['negation'] = ev_rows == [r[2:7] for r in loader_l4b['negation']]
        if 'added_poles' in loader_l4b:
            from inspect_checkpoint import added_poles as ap_ev
            from eval_greybox import build as ev_build
            cfg = run_cfg(os.path.join(STAGING, str(rid)))
            data, norm, fs = ev_build(cfg)
            from gantry_dynamic.training import load_checkpoint
            load_checkpoint(fs, os.path.join(STAGING, str(rid), 'gantry_ckpt_%d' % rid),
                            joint_estimation=True, start_phase='lbfgs')
            rep['added_poles'] = [float(v) for v in ap_ev(fs, data, cfg.nx_phys, cfg.nx_ann)] == loader_l4b['added_poles']
    return rep


def main():
    from gantry_dynamic.training import load_checkpoint
    H.assert_roots()
    runs = discover(STAGING)
    parts = args.parts.split(',')
    if args.eval_check and args.truncate_s is not None:
        # LOCAL SCRIPT TEST ONLY: the evaluation at the loader's short records (its run_config forces
        # full length; the staged configs must then record TRUNCATE_S null, as eval_common requires)
        from dataclasses import replace
        import eval_greybox
        import eval_overlap
        import eval_negation
        full = eval_greybox.run_config
        for m in (eval_greybox, eval_overlap, eval_negation):
            m.run_config = lambda run_dir: replace(full(run_dir), truncate_s=args.truncate_s)
    # H2: the clean source manifest (both sides, provenance); compare_l4.py refuses a clean output
    # whose manifest differs from the current Clean/ tree
    summary = {'side': args.side, 'staging': STAGING, 'runs': runs, 'lstsq_driver': LSTSQ_DRIVER,
               'clean_tree': H.clean_tree_hashes(), 'checks': {}, 'eval_check': {}}
    for run in args.runs.split(','):
        rid = runs[run]
        run_dir = os.path.join(STAGING, str(rid))
        t0 = time.time()
        cfg = run_cfg(run_dir)
        data, norm, fs = build(cfg)
        base = os.path.join(run_dir, 'gantry_ckpt_%d' % rid)
        load_checkpoint(fs, base, joint_estimation=True, start_phase='lbfgs')
        ref_score = float(np.load(base + '.npz', allow_pickle=True)['bestfit'])
        print('[%s %d] built and loaded in %.0f s' % (run, rid, time.time() - t0), flush=True)
        if 'l4' in parts:
            summary['checks'][run] = l4(run, rid, cfg, data, norm, fs, ref_score)
        res = l4b(run, rid, cfg, data, norm, fs) if 'l4b' in parts else {}
        if args.eval_check:
            summary['eval_check'][run] = eval_check(run, rid, res, parts)
            print('[%s %d] eval check %s' % (run, rid, summary['eval_check'][run]), flush=True)
        with open(os.path.join(OUT, 'summary.json'), 'w') as fh:
            json.dump(summary, fh, indent=1, default=str)
    H.assert_roots()
    bad = [r for r, c in summary['checks'].items() if not c.get('val_score', {}).get('ok', True)]
    bad += ['eval:%s:%s' % (r, k) for r, d in summary['eval_check'].items() for k, v in d.items() if not v]
    print('L4_LOADER_DONE side=%s problems=%s' % (args.side, bad or 'none'), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    if args.side == 'mainline':
        os.environ.setdefault('THESIS_EVAL', 'final')
        os.environ['THESIS_RUNS_DIR'] = STAGING
        sys.path.insert(0, os.path.join(H.REPO, 'Thesis-writeup', 'Code', 'Evaluation'))
    main()
