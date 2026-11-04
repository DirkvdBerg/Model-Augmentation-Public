"""D-295: one training phase of a thesis run, started from its saved state, on one side, in one process.

    python -u phase_run.py --side S --phase adam|polish --run R1 --staging DIR --dumps DIR --out DIR [--n-its N]

adam    resume Adam from the thesis deepSI `_last.pth` (DUMPS/SSE_Interconnect_Composed_<job>_last.pth) for N
        updates (one epoch past the dump's, capped at N), then a two-iteration polish so the job ends as a run does.
polish  the full L-BFGS polish of the thesis run from its `_pre_lbfgs` pair (STAGING/<job>/), with the thesis
        polish settings of its config.json.
Every other setting is the thesis run's: mainline builds it from the row (hcommon.make_cfg) with the thesis run
environment of the staged config.json; Clean/ reads the staged folder through clean-up-verification/convert_run.py.
The clean side of adam first runs extract_dump.py (mainline root) on the dump. Mainline runs unmodified, as in
run_l3bc.py layer 3c (save_dir and the post-run report replaced, no number touched).

Writes OUT/: run/ (the run folder: both checkpoint pairs, val_hist.npz), preds.npz (the final model's closed-loop
prediction, y, sigma_e and record names on val and test, full records), result.json (final bestfit, polish
outcome, epochs, the Clean/ tree state final|reverted, settings that differ from the staged config.json), meta.json.
Compare with compare_phase.py.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import hcommon as H

# the final thesis runs: preset -> (harness row, job id), as l4_stage.py
RUNS = {H.ROWS[r][0]: (r, j) for r, j in {221: 89792, 222: 89796, 223: 89797, 224: 89793, 225: 89798, 226: 89799,
                                         227: 89800, 229: 89794, 230: 89795}.items()}
# config.json keys of the run environment and per-run values; the rest must match the staged thesis config.json
ENV_KEYS = {'DEVICE', 'COMPILE_MODE', 'TRUNCATE_S', 'EPOCHS', 'N_ITS', 'ITS_PER_VAL', 'LBFGS_MAX_ITER',
            'LBFGS_INNER_ITER', 'LBFGS_WINDOWS', 'GIT', 'NF_PROBE_PRINT', 'START_PHASE', 'run_id', 'data_dir',
            'RESUMES', 'CONVERTED_FROM'}

ap = argparse.ArgumentParser()
ap.add_argument('--side', required=True, choices=('mainline', 'clean'))
ap.add_argument('--phase', required=True, choices=('adam', 'polish'))
ap.add_argument('--run', required=True, choices=sorted(RUNS))
ap.add_argument('--staging', required=True, help='the l4_stage.py folder: <job>/ with config.json and both pairs')
ap.add_argument('--dumps', default=None, help='folder with the thesis deepSI dumps (adam)')
ap.add_argument('--out', required=True)
ap.add_argument('--n-its', type=int, default=50, help='adam: Adam updates after the resume')
ap.add_argument('--zero-moments', action='store_true',
                help='adam, clean side: resume from zero Adam moments, as mainline does from a deepSI dump')
ap.add_argument('--predict-truncate-s', type=float, default=None,
                help='LOCAL SCRIPT TEST ONLY: predict on records cut to this length instead of the full records')
args = ap.parse_args()
row, job = RUNS[args.run]
OUT = os.path.abspath(args.out)
if os.path.exists(OUT):
    sys.exit('%s exists; remove it first' % OUT)
os.makedirs(OUT)
lad = os.path.join(OUT, 'localappdata')
os.makedirs(os.path.join(lad, 'deepSI', 'checkpoints'))
os.environ['LOCALAPPDATA'] = lad
staged = os.path.join(os.path.abspath(args.staging), str(job))
with open(os.path.join(staged, 'config.json'), encoding='utf-8') as fh:
    thesis_cfg = json.load(fh)
dump = None
if args.phase == 'adam':
    if not args.dumps:
        sys.exit('--dumps is required for the adam phase')
    dump = os.path.join(os.path.abspath(args.dumps), 'SSE_Interconnect_Composed_%d_last.pth' % job)
    if not os.path.exists(dump):
        sys.exit('thesis dump %s not found' % dump)
run_id = 'P-%s-%s' % (args.phase, args.run)
run_dir = os.path.join(OUT, 'run')
T0 = time.time()

if args.side == 'clean' and dump is not None:
    # the dump's state as plain tensors, read on the mainline root in its own process
    plain = os.path.join(OUT, 'dump_state.pt')
    subprocess.run([sys.executable, '-u', os.path.join(H.HERE, 'extract_dump.py'), dump, plain], check=True)

H.select_side(args.side)
sys.path.append(os.path.join(H.REPO, 'clean-up-verification'))
import gantry_dynamic  # noqa: E402,F401  (mainline: puts model_augmentation on sys.path before a dump is unpickled)
import numpy as np  # noqa: E402
import torch  # noqa: E402


def tree_state():
    """final, reverted or mixed: which side of the D-294 swaps (d294_revert.SWAPS) the imported Clean/ is on."""
    from d294_revert import SWAPS
    got = set()
    for rel, final, reverted in SWAPS:
        text = open(os.path.join(H.CLEAN, *rel.split('/')), encoding='utf-8').read().splitlines()
        got.add('final' if final in text and reverted not in text else
                'reverted' if reverted in text and final not in text else 'mixed')
    return got.pop() if len(got) == 1 else 'mixed'


# 1. Budget: the thesis run environment; adam caps Adam at N updates one epoch past the dump and polishes briefly.
env = dict(truncate_s=thesis_cfg['TRUNCATE_S'], epochs=thesis_cfg['EPOCHS'], n_its=thesis_cfg.get('N_ITS'),
           its_per_val=thesis_cfg['ITS_PER_VAL'], lbfgs_max_iter=thesis_cfg['LBFGS_MAX_ITER'],
           lbfgs_inner_iter=thesis_cfg['LBFGS_INNER_ITER'], lbfgs_windows=thesis_cfg['LBFGS_WINDOWS'])
done = None
if dump is not None:
    d = torch.load(os.path.join(OUT, 'dump_state.pt') if args.side == 'clean' else dump, map_location='cpu',
                   weights_only=False)
    done = int(round(float(d.get('epochs_before', 0)) + float(d['epoch_counter'])))
    del d
    env.update(epochs=done + 1, n_its=args.n_its, its_per_val=args.n_its, lbfgs_max_iter=2, lbfgs_inner_iter=1,
               lbfgs_windows=256)

# 2. Run the phase.
if args.side == 'mainline':
    src_dir = os.path.join(OUT, 'source')
    os.makedirs(src_dir)
    if args.phase == 'polish':
        src = 'gantry_ckpt_%d_pre_lbfgs' % job
        for ext in ('.pt', '.npz'):
            shutil.copy2(os.path.join(staged, src + ext), src_dir)
    else:
        src = os.path.basename(dump)
        shutil.copy2(dump, src_dir)
    os.environ['RESUME_CHECKPOINT'] = os.path.join(src_dir, src)
    extra = dict(start_phase='lbfgs') if args.phase == 'polish' else {}
    cfg, EP = H.make_cfg(row, 'lowpass', run_id=run_id, budget=env, **extra)
    H.assert_roots()
    EP.save_dir = lambda cfg: OUT
    EP.evaluate_and_save = lambda *a, **k: None
    EP.state_recovery_diagnostic = lambda *a, **k: None
    EP.compute_gradient_norms = lambda *a, **k: ({}, {})
    EP.main()
    os.rename(os.path.join(OUT, run_id), run_dir)
    final_base = os.path.join(run_dir, 'gantry_ckpt_%s' % run_id)
    from gantry_dynamic.config import config_json_dict
    written = json.loads(json.dumps(config_json_dict(cfg), default=str))
else:
    from convert_run import convert_dump, convert_run
    from gantry_dynamic.config import RunConfig
    import gantry_interconnect_dynamic as EP
    convert_run(staged, run_dir)
    clean_id = job
    if dump is not None:
        # deepSI keeps its checkpoints in one folder per user on Linux (~/.deepSI, LOCALAPPDATA on Windows), where
        # both arms of a run resume at the same time: each gets its own run id, its pairs renamed to match
        clean_id = '%s-%s%s' % (run_id, tree_state(), '-zero' if args.zero_moments else '')
        cj_path = os.path.join(run_dir, 'config.json')
        with open(cj_path, encoding='utf-8') as fh:
            cj = json.load(fh)
        cj['run_id'] = clean_id
        with open(cj_path, 'w') as fh:
            json.dump(cj, fh, indent=2)
        for f in os.listdir(run_dir):
            if f.startswith('gantry_ckpt_%d' % job):
                os.rename(os.path.join(run_dir, f), os.path.join(run_dir, f.replace(str(job), clean_id, 1)))
        convert_dump(os.path.join(OUT, 'dump_state.pt'), EP._deepsi_checkpoint_paths(clean_id)[1],
                     zero_moments=args.zero_moments)
    os.environ['SLURM_JOB_ID'] = run_id                  # the rerun id of the previous/ folder
    with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as fh:
        meta0 = json.load(fh)
    cfg = RunConfig.from_config_json(meta0, device='cpu', compile_mode=None, **H.clean_inner_iter(dict(env)))
    H.assert_roots()
    try:
        EP.main(cfg, rerun=('resume', run_dir, 'last') if args.phase == 'adam' else ('polish-only', run_dir, None),
                evaluate_run=False)
    finally:
        # the deepSI checkpoints this job wrote belong to the task folder, not to the user's deepSI folder
        if dump is not None:
            os.makedirs(os.path.join(OUT, 'deepsi'), exist_ok=True)
            for p in EP._deepsi_checkpoint_paths(clean_id):
                if os.path.exists(p):
                    shutil.move(p, os.path.join(OUT, 'deepsi', os.path.basename(p)))
    final_base = os.path.join(run_dir, 'gantry_ckpt_%s' % clean_id)
    from convert_run import thesis_settings
    written = thesis_settings(cfg)
H.assert_roots()
seconds_phase = time.time() - T0


# 3. The final model's closed-loop prediction on val and test, full records (l4_loader.l4, model rows only).
def predict(cfg):
    from dataclasses import replace
    from gantry_dynamic.data import load_datasets, compute_normalization, servo_sigma
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop, build_controller_bank
    from gantry_dynamic.training import load_checkpoint
    from model_augmentation.fit_systems.closed_loop import closed_loop_free_run_rms_batch
    clean = args.side == 'clean'
    cfg = replace(cfg, truncate_s=args.predict_truncate_s)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data = load_datasets(cfg); norm = compute_normalization(cfg, data)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    fs = build_model(cfg, data, norm) if clean else build_model(cfg.hp, cfg, data, norm)
    vd = {'controllers': data.controllers} if clean else {'val_data': data.val_ckpt_data}
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=data.train_files, val_files=data.val_files,
                                     verbose=False, e_train=data.e_train, e_val=data.e_val, **vd)
    if clean:
        load_checkpoint(fs, final_base, polish_only=True)
        from gantry_dynamic.controller import design_y_and_gain
        y_op_for = lambda f, g: design_y_and_gain(f, data.controllers[f], g)              # noqa: E731
        refused, gain, ctrl = ValueError, False, (data.controllers,)
    else:
        load_checkpoint(fs, final_base, cfg.joint_estimation, start_phase='lbfgs')
        from gantry_dynamic.controller import y_op_for
        refused, gain, ctrl = KeyError, cfg.controller_gain, ()
    out = {}
    for split in ('val', 'test'):
        files, sdl, es = [], [], []
        for f, sd, e in zip(getattr(data, split + '_files'), getattr(data, split + '_list'),
                            getattr(data, 'e_' + split)):
            try:
                y_op_for(f, gain)
            except refused:
                continue
            files.append(f); sdl.append(sd); es.append(e)
        bank, rows, _ = build_controller_bank(files, *ctrl, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u,
                                              dtype=cfg.dtype_pt, use_gain=gain)
        k0 = max(fs.na, fs.nb)
        with torch.no_grad():
            _, _, preds = closed_loop_free_run_rms_batch(fs, sdl, bank, rows, return_pred=True)
        out[split + '_records'] = np.array([os.path.basename(str(f).replace('\\', '/')).split('_')[0]
                                            for f in files])
        out[split + '_y'] = np.stack([np.asarray(sd.y, float)[k0:] for sd in sdl])
        out[split + '_sigma_e'] = servo_sigma(es, k0)
        out[split + '_model'] = np.stack([np.asarray(p, float) for p in preds])
    return out


np.savez_compressed(os.path.join(OUT, 'preds.npz'), **predict(cfg))
H.assert_roots()

# 4. result.json and meta.json.
with np.load(final_base + '.npz', allow_pickle=True) as z:
    outcome = json.loads(str(z['lbfgs_outcome']))
    bestfit, done_after = float(z['bestfit']), int(z['done_epochs'])
differs = {k: (thesis_cfg[k], written[k]) for k in sorted(set(thesis_cfg) & set(written))
           if k not in ENV_KEYS and thesis_cfg[k] != written[k]}
result = dict(side=args.side, phase=args.phase, run=args.run, job=job, run_id=run_id, final_base=final_base,
              bestfit=bestfit, lbfgs_outcome=outcome, dump=dump, dump_epochs=done, done_epochs=done_after,
              n_its=args.n_its if dump else None, zero_moments=args.zero_moments, budget=env,
              settings_differ_from_thesis=differs,
              clean_tree=tree_state() if args.side == 'clean' else None,
              seconds_phase=seconds_phase, seconds=time.time() - T0)
with open(os.path.join(OUT, 'result.json'), 'w') as fh:
    json.dump(result, fh, indent=1, default=str)
import gantry_dynamic.data as GDD  # noqa: E402


class _D:
    train_files, val_files, test_files = GDD.record_files(cfg)


H.write_meta(OUT, cfg, _D, dict(case=run_id, layer='D-295 ' + args.phase))
print('PHASE_RUN_DONE side=%s phase=%s run=%s bestfit=%.6e accepted=%s differs=%s seconds=%.0f'
      % (args.side, args.phase, args.run, bestfit, outcome.get('accepted') if outcome else None,
         sorted(differs) or 'none', time.time() - T0), flush=True)
