"""Predictions of the selected black box (R8) and the floors on every split.

Black box: deepSI `SS_encoder_general_hf` on the closed-loop map (r, f) -> e = y - r, loaded from
its `_best.pth` (the checkpoint selected on validation), fed exactly as in its training script
(`train_bb_clmap.load`: --target e --in64 --f-blockmean), y_hat = r + e_hat, scored from the run's
k0 (result.json; 29 in the 2026-10-11 search, 30 in the final one), on every split and the twins.
E5 is left out: the black box models the K1 loop, so a K2 record is outside its definition.
Check: the validation RMS must reproduce `result.json` `per_record`.

Floor per record of every split: the noisy record minus its noise-free twin, both through the same
lowpass-mode loader (D-232 decimation and trim), from k0 (THESIS-RESULTS section 2).
Also stored per record: sigma_e, the servo-error std over the same samples, for NRMS_e (D-262).
Output: preds_BB.npz (test, the name the figures read) and preds_BB_<split>.npz.
Selection (final campaign, THESIS_BB_DIR set): the grid's rule is applied to every configuration of
the grid job first (select_bb); a THESIS_BB_DIR other than the selected one stops the evaluation.

Usage: python Thesis-writeup/Code/Evaluation/eval_blackbox.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from eval_common import (REPO, TW, BB_CKPT, BB_CODE, BB_RESULT, CACHE, DATA_NF, TWINS_NOISY, TWINS_NF,
                         SPLITS, record_name, twin_files)

sys.path.insert(0, str(REPO / 'scripts' / 'gantry' / 'blackbox-cl-trainable' / 'bbcl'))


def load_records(cfg, files, folder=None):
    """(u, target e, r, y) per record, as train_bb_clmap.load with --target e --in64 --f-blockmean."""
    from gantry_dynamic.data import traj_dir, _record_at_fs_new, _load_u, _resample_u
    from gantry_dynamic.config import record_trim
    out = []
    for fn in files:
        d = loadmat(os.path.join(folder or traj_dir(cfg), fn), squeeze_me=True)
        _, y, r, f = _record_at_fs_new(d, cfg, ('r_sim', 'f_sim'))
        r, y = np.asarray(np.broadcast_to(r, y.shape), float), np.asarray(y, float)
        n = len(_resample_u(_load_u(d), cfg))
        f = _resample_u(np.asarray(d['f_sim'], float), cfg)[:n]
        k = record_trim(cfg)
        f = f[k:n - k] if k else f
        assert f.shape == y.shape, (fn, f.shape, y.shape)
        out.append((np.hstack([r, f]), y - r, r, y))
    return out


def select_bb(bb_dir):
    """The R8 selection rule (thesis_r8_grid.conf FINAL FORM, D-264), applied to every configuration
    of the grid job that `bb_dir` belongs to (sibling folders `<index>_<name>_n<its>_<job>`): lowest
    best-validation NRMS_e (mean over the validation records and axes of the reloaded best
    checkpoint, result.json `nrms_e_per_record`); within 2 % of it, the configuration with fewer
    parameters (counted in its `_best.pth`). Writes Results/res_bb_selection.csv (the appendix search
    table) and refuses a THESIS_BB_DIR that is not the selected configuration."""
    bb_dir = Path(bb_dir).resolve()
    job = bb_dir.name.rsplit('_', 1)[-1]
    rows = []
    for d in sorted(bb_dir.parent.glob('*_%s' % job), key=lambda p: int(p.name.split('_')[0])):
        if not (d / 'result.json').exists():
            print('[BB select] %s: no result.json, left out' % d.name, flush=True)
            continue
        res = json.load(open(d / 'result.json'))
        score = float(np.mean(res['nrms_e_per_record']))
        hist = [h['nrms_e'] for h in res.get('val_rms_hist') or [] if np.isfinite(h['nrms_e'])]
        ckd = d / 'localappdata' / 'deepSI' / 'checkpoints'
        # result.json names the file since 2026-10-16; earlier searches hold one `_best.pth`
        ck = ckd / res['checkpoint'] if 'checkpoint' in res else next(iter(sorted(ckd.glob('*_best.pth'))), ckd)
        n_par = np.nan
        if ck.is_file():
            sd = torch.load(ck, map_location='cpu', weights_only=False)
            n_par = sum(p.numel() for m in sd.values() if isinstance(m, torch.nn.Module) for p in m.parameters())
        parts = d.name.split('_')
        a = res['args']
        rows.append(dict(index=int(parts[0]), name='_'.join(parts[1:-2]), dir=d, nx=a['nx'], width=a['fh_width'],
                         lr=a['lr'], score=score, hist_min=min(hist) if hist else np.nan, n_par=n_par,
                         best_update=res.get('best_update'), status=res.get('status', '')))
    ok = [r for r in rows if np.isfinite(r['score'])]
    best = min(r['score'] for r in ok)
    # HEURISTIC: the grid's stated rule (thesis_r8_grid.conf SELECTION, D-264), 2 % band
    near = [r for r in ok if r['score'] <= 1.02 * best]
    sel = min(near, key=lambda r: (r['n_par'] if np.isfinite(r['n_par']) else np.inf, r['score']))
    for r in rows:
        print('[BB select] %2d %-16s nx %2d w %2d lr %.0e | best val NRMS_e %.4f (history min %.4f) | %s params | '
              'update %s | %s%s' % (r['index'], r['name'], r['nx'], r['width'], r['lr'], r['score'], r['hist_min'],
                                    r['n_par'], r['best_update'], r['status'], '  <- SELECTED' if r is sel else ''),
              flush=True)
    path = TW / 'Results' / 'res_bb_selection.csv'
    with open(path, 'w') as fh:
        fh.write('index,name,nx,width,lr,best_val_nrms_e,n_params,best_update,status,within_2pct,selected\n')
        for r in rows:
            fh.write('%d,%s,%d,%d,%.0e,%.6f,%s,%s,%s,%d,%d\n' % (
                r['index'], r['name'], r['nx'], r['width'], r['lr'], r['score'], r['n_par'], r['best_update'],
                r['status'], int(r in near), int(r is sel)))
    print('wrote', path)
    if sel['dir'] != bb_dir:
        raise ValueError('THESIS_BB_DIR is %s, the selection rule picks %s' % (bb_dir.name, sel['dir'].name))
    print('[BB select] THESIS_BB_DIR %s is the selected configuration' % bb_dir.name, flush=True)


def main():
    import deepSI
    from deepSI.fit_systems.encoders import SS_encoder_general_hf
    from setup import cfg_for
    from gantry_dynamic.data import record_files
    if os.environ.get('THESIS_BB_DIR'):
        select_bb(os.environ['THESIS_BB_DIR'])
    t0 = time.time()
    from gantry_dynamic.data import servo_sigma
    cfg = cfg_for('lowpass', 1, local=True)
    tr_f, va_f, te_f = record_files(cfg)
    te_f = [f for f in te_f if record_name(f) != 'E5']

    res = json.load(open(BB_RESULT))
    # the file name the run recorded (deepSI names it after the class; audit 2026-10-16), else the
    # name of the 2026-10-11 search, which predates the field
    ckpt = res.get('checkpoint', 'SS_encoder_general_hf_%s_best.pth' % BB_CODE)
    model = SS_encoder_general_hf.__new__(SS_encoder_general_hf)
    model.__dict__ = torch.load(BB_CKPT / ckpt, map_location='cpu', weights_only=False)
    print('[BB] nx %d, loaded in %.0f s' % (model.nx, time.time() - t0), flush=True)

    def predict(recs):
        preds = []
        for u, e, r, y in recs:
            sd = deepSI.System_data(u=np.ascontiguousarray(u), y=np.ascontiguousarray(e), dt=cfg.ts_new)
            with np.errstate(all='ignore'):
                preds.append(r + np.asarray(model.apply_experiment(sd).y, float))
        return preds

    k0 = int(res['k0'])                                   # the run's own scored interval (D-250)
    files = {'train': (tr_f, None, str(DATA_NF)), 'val': (va_f, None, str(DATA_NF)),
             'test': (te_f, None, str(DATA_NF)),
             'twins': (twin_files(), str(TWINS_NOISY), str(TWINS_NF))}
    for split in [sp for sp in SPLITS if sp in files]:
        fl, folder, folder_nf = files[split]
        recs = load_records(cfg, fl, folder=folder)
        pr = predict(recs)
        if split == 'val':                                # check: per-record RMS of result.json
            mine = np.array([np.sqrt(np.mean((p[k0:] - y[k0:]) ** 2, 0)) for p, (_, _, _, y) in zip(pr, recs)])
            rel = np.max(np.abs(mine - np.array(res['per_record'])) / np.array(res['per_record']))
            print('[BB] check validation per_record vs result.json: max rel. difference %.1e' % rel, flush=True)
            assert rel < 1e-4, 'black box does not reproduce its validation score'
        out = dict(records=np.array([record_name(f) for f in fl]), k0=k0, split=split,
                   y=np.stack([y[k0:] for _, _, _, y in recs]),
                   r=np.stack([r[k0:] for _, _, r, _ in recs]),
                   sigma_e=servo_sigma([y - r for _, _, r, y in recs], k0),
                   bb=np.stack([p[k0:] for p in pr]))
        if folder_nf is not None:                         # floor: noisy minus noise-free twin
            nf = load_records(cfg, fl, folder=folder_nf)
            out['y_nf'] = np.stack([y[k0:] for _, _, _, y in nf])
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE / ('preds_BB.npz' if split == 'test' else 'preds_BB_%s.npz' % split),
                            code=BB_CODE, **out)
        rms = np.sqrt(np.mean((out['bb'] - out['y']) ** 2, axis=1))
        for n, a, sg in zip(out['records'], rms, out['sigma_e']):
            print('  %-5s %-4s BB %s | NRMS_e %s' % (split, n, ' '.join('%.3e' % v for v in a),
                                                     ' '.join('%.3f' % v for v in a / sg)))
    print('[BB] done in %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
