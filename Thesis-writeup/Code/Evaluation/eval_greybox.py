"""Closed-loop test-set predictions of one grey-box run (THESIS-RESULTS section 5).

For run R (job id from eval_common.RUNS) it loads the final model (`gantry_ckpt_<id>`, the model
the run's own report scored), rebuilds the K1 controller bank for the test records the report
scored (E5 is refused by the scorer, D-227), and stores per record, from k0:
  y         the measured (noisy) output
  model     the trained model
  base_true the physics block at the true combinations, learned block off (report row 4)
  clamped   the model with every output to an added state disabled (only when n_a > 0)
Check: the model RMS must reproduce the run's saved `cl_test_rms` row 0.

Usage: python Thesis-writeup/Code/Evaluation/eval_greybox.py R5 [R3 ...]
"""
import sys
import time

import numpy as np
import torch

from eval_common import RUNS, RUNS_DIR, CACHE, K0, record_name


def run_config(run_dir):
    """The run's RunConfig from its config.json, as inspect_checkpoint.checkpoint_config does, but
    without its required-key list: that list names AUGMENTATION_INIT, which RunConfig no longer has
    and the 2026-10-11 runs therefore do not write."""
    import json
    from dataclasses import fields
    from gantry_dynamic.config import RunConfig
    meta = json.load(open(run_dir / 'config.json', encoding='utf-8'))
    opts = {f.name: meta[f.name.upper()] for f in fields(RunConfig) if f.name.upper() in meta}
    for name in ('ann_route_ix', 'lbfgs_freeze'):
        if name in opts:
            opts[name] = tuple(opts[name])
    opts.update(device='cpu', compile_mode=None, truncate_s=None,
                na_nb_override=int(meta['NA_NB']), nf_override=int(meta['NF']))
    return RunConfig(**opts)


def build(cfg):
    """Data, normalisation and model as inspect_checkpoint.thesis_model, from a given cfg."""
    from gantry_dynamic.data import load_datasets, compute_normalization
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data = load_datasets(cfg); norm = compute_normalization(cfg, data)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    fs = build_model(cfg.hp, cfg, data, norm)
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=data.train_files, val_files=data.val_files,
                                     val_data=data.val_ckpt_data, verbose=False)
    return data, norm, fs


def frozen_y(m, y_op=0.0):
    """Step 1 LTI baseline: the same physics block with the scheduling frozen at Y = y_op.

    Uses the block's own frozen path (`Y_op`, `N_op`, `d_op` in `_deriv_with`): M(Y)^{-1} is
    evaluated once at y_op from the current matrices, so the model is the LPV baseline with
    Y removed from the scheduling and nothing else changed. Only y_op = 0 (mid-stroke) is used.
    """
    phy = next(b for b in m.hfn.connected_blocks if hasattr(b, 'free_params'))
    with torch.no_grad():
        K, C, A, mh, alpha, beta, gamma_, N0, N1, N2 = phy.pass_mats()
        assert y_op == 0.0
        phy.N_op = N0.detach().clone()                     # N(0) = N0
        phy.d_op = (mh * (alpha * gamma_ - beta ** 2)).detach().clone()   # d(0)
        phy.Y_op = float(y_op)
    return m


def evaluate(run, lti_only=False, only=('base_true', 'clamped')):
    from inspect_checkpoint import clamped
    from gantry_dynamic.training import load_checkpoint
    from gantry_dynamic.controller import build_controller_bank, y_op_for
    from gantry_dynamic.closed_loop_report import _without_learned, _true_free
    from model_augmentation.fit_systems.closed_loop import closed_loop_free_run_rms_batch

    rid = RUNS[run]
    base = str(RUNS_DIR / str(rid) / ('gantry_ckpt_%d' % rid))
    t0 = time.time()
    cfg = run_config(RUNS_DIR / str(rid))
    data, norm, fs = build(cfg)
    load_checkpoint(fs, base, joint_estimation=True, start_phase='lbfgs')
    fs._sync_prediction_state_for_validation(refresh_basis=False)
    print('[%s %d] built and loaded in %.0f s' % (run, rid, time.time() - t0), flush=True)

    files, sdl = [], []
    for f, sd in zip(data.test_files, data.test_list):
        try:
            y_op_for(f, cfg.controller_gain)
        except KeyError:
            print('  skipped %s (controller gain, D-227)' % record_name(f))
            continue
        files.append(f)
        sdl.append(sd)
    bank, rows, _ = build_controller_bank(files, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u,
                                          dtype=cfg.dtype_pt, use_gain=cfg.controller_gain)
    assert max(fs.na, fs.nb) == K0

    phy = next(b for b in fs.hfn.connected_blocks if hasattr(b, 'free_params'))
    if lti_only:
        prev = dict(np.load(CACHE / ('preds_%s.npz' % run), allow_pickle=True))
        systems = {'base_lti': frozen_y(_without_learned(fs, _true_free(phy)))}
    else:
        systems = {'model': fs}
        if 'base_true' in only:
            systems['base_true'] = _without_learned(fs, _true_free(phy))
        if cfg.nx_ann and 'clamped' in only:
            systems['clamped'] = clamped(fs, cfg.ann_route_ix, cfg.nx_phys)
    out = {'records': np.array([record_name(f) for f in files]),
           'y': np.stack([np.asarray(sd.y, float)[K0:] for sd in sdl])}
    for tag, s in systems.items():
        t1 = time.time()
        with torch.no_grad():
            _, _, preds = closed_loop_free_run_rms_batch(s, sdl, bank, rows, return_pred=True)
        out[tag] = np.stack([np.asarray(p, float) for p in preds])
        print('  %-9s rollout %.0f s' % (tag, time.time() - t1), flush=True)

    if lti_only:
        assert list(prev['records']) == list(out['records'])
        prev['base_lti'] = out['base_lti']
        np.savez_compressed(CACHE / ('preds_%s.npz' % run), **prev)
        mine = np.sqrt(np.mean((out['base_lti'] - out['y']) ** 2, axis=1))
        lpv = np.sqrt(np.mean((prev['base_true'] - out['y']) ** 2, axis=1))
        for n, a, b in zip(out['records'], mine, lpv):
            print('  %-3s LTI %s | LPV %s' % (n, ' '.join('%.3e' % v for v in a), ' '.join('%.3e' % v for v in b)))
        print('[%s %d] LTI added in %.0f s' % (run, rid, time.time() - t0), flush=True)
        return

    # check against the run's own report (row 0 model, row 3 baseline true)
    saved = np.load(RUNS_DIR / str(rid) / ('gantry_results_%d.npz' % rid), allow_pickle=True)
    names = [record_name(f) for f in saved['cl_test_files']]
    for tag, row in (('model', 0), ('base_true', 3)):
        if tag not in out:
            continue
        mine = np.sqrt(np.mean((out[tag] - out['y']) ** 2, axis=1))
        ref = np.stack([saved['cl_test_rms'][row, names.index(r)] for r in out['records']])
        rel = np.max(np.abs(mine - ref) / ref)
        print('  check %-9s max rel. difference to the run report %.1e' % (tag, rel), flush=True)
        assert rel < 1e-6, 'does not reproduce the run report'
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / ('preds_%s.npz' % run), run=run, job=rid, nx_ann=cfg.nx_ann,
                        obc=cfg.obc, ps2=bool(getattr(cfg, 'ps2', False)), **out)
    print('[%s %d] done in %.0f s' % (run, rid, time.time() - t0), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    # R3:clamped -> model + clamped rollouts; R6 -> model only; R5 --lti -> add the LTI baseline
    lti = '--lti' in sys.argv
    for a in [a for a in sys.argv[1:] if a != '--lti']:
        run, _, extra = a.partition(':')
        evaluate(run, lti_only=lti, only=tuple(e for e in extra.split(',') if e))
