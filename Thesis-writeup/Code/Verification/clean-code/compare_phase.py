"""D-295: the checks on phase_run.py outputs; each prints one verdict line and writes a JSON report.

    python -u compare_phase.py bitwise MAINLINE_OUT CLEAN_OUT REPORT.json
        arm A: the final and pre-polish pairs (hfn as Clean/ names it, encoder, Adam state), bestfit, the polish
        decision and scores, the validation scores and the predictions on val and test, bitwise. The clean output
        must come from the reverted tree and record the current Clean/ manifest (H2).
    python -u compare_phase.py thesis OUT STAGING REPORT.json
        a polish rerun against the thesis final pair: the same accept decision (a row l4_stage.py staged as the
        D-274 rebuilt model expects a rejection) and bestfit within 1e-3 relative, the existing CPU-eager vs
        GPU-compiled acceptance (l4_loader.l4).
    python -u compare_phase.py floor A_OUT B_OUT CSV_DIR REPORT.json
        arm B against arm A, both clean, the same phase: per val/test record and axis, the RMS of the difference of
        the two models' predictions must be below the noise floor RMS of that record (the `floor` rows of
        res_errors.csv: the noisy record minus its noise-free twin).
"""
import csv
import json
import os
import sys

import numpy as np

import hcommon as H

sys.path.append(os.path.join(H.REPO, 'clean-up-verification'))


def _load(out):
    with open(os.path.join(out, 'result.json')) as fh:
        return json.load(fh)


def _same(a, b):
    import torch
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _pair(base, mainline):
    """(hfn, encoder, Adam state or None) of a pair; a mainline hfn under the Clean/ names."""
    import torch
    from convert_run import convert_hfn_state
    ck = torch.load(base + '.pt', map_location='cpu')
    hfn = convert_hfn_state(ck['hfn'])[0] if mainline else ck['hfn']
    return hfn, ck['encoder'], ck.get('optimizer')


def _finish(report, path, label):
    with open(path, 'w') as fh:
        json.dump(report, fh, indent=1, default=str)
    print('COMPARE_PHASE %s %s %s' % (label, report['verdict'], json.dumps(report.get('problems', []))[:2000]))
    sys.exit(0 if report['verdict'] == 'PASS' else 1)


def bitwise(main_out, clean_out, path):
    rm, rc = _load(main_out), _load(clean_out)
    label = 'bitwise %s %s' % (rc['phase'], rc['run'])
    probs = []
    if rc['clean_tree'] != 'reverted':
        probs.append('clean output from the %s tree, not the reverted one (arm A)' % rc['clean_tree'])
    with open(os.path.join(clean_out, 'meta.json')) as fh:
        probs += H.clean_tree_problems(json.load(fh)['sources']['clean'])
    if probs:
        _finish(dict(verdict='REFUSED', problems=probs), path, label)
    checks = {}
    for tag, mb, cb in (('final', rm['final_base'], rc['final_base']),
                        ('pre_lbfgs', rm['final_base'] + '_pre_lbfgs', rc['final_base'] + '_pre_lbfgs')):
        m, c = _pair(mb, True), _pair(cb, False)
        checks[tag + '_hfn'] = _same(m[0], c[0])
        checks[tag + '_encoder'] = _same(m[1], c[1])
        # the pre-polish pair: mainline stores the Adam moments in it, Clean/ never does (nothing resumes Adam from it)
        if tag == 'final':
            checks[tag + '_adam'] = ((m[2] is None) == (c[2] is None)
                                     and (m[2] is None or _same(m[2]['state'], c[2]['state'])))
    checks['bestfit'] = rm['bestfit'] == rc['bestfit']
    om, oc = rm['lbfgs_outcome'] or {}, rc['lbfgs_outcome'] or {}
    for k in ('ran', 'accepted', 'val_before', 'val_after'):
        checks['polish_' + k] = om.get(k) == oc.get(k)
    vm, vc = (np.load(os.path.join(o, 'run', 'val_hist.npz'), allow_pickle=True) for o in (main_out, clean_out))
    checks['val_hist_score'] = vm['val_hist_score'].tobytes() == vc['val_hist_score'].tobytes()
    pm, pc = (np.load(os.path.join(o, 'preds.npz')) for o in (main_out, clean_out))
    for k in pm.files:
        checks['preds_' + k] = k in pc.files and pm[k].dtype == pc[k].dtype and pm[k].shape == pc[k].shape and (
            pm[k].tobytes() == pc[k].tobytes())
    checks['dump_epochs'] = rm['dump_epochs'] == rc['dump_epochs']
    probs = [k for k, ok in checks.items() if not ok]
    _finish(dict(verdict='FAIL' if probs else 'PASS', problems=probs, checks=checks,
                 bestfit=(rm['bestfit'], rc['bestfit']), accepted=(om.get('accepted'), oc.get('accepted')),
                 differs_from_thesis=(rm['settings_differ_from_thesis'], rc['settings_differ_from_thesis'])),
            path, label)


def thesis(out, staging, path):
    r = _load(out)
    label = 'thesis %s %s %s' % (r['phase'], r['run'], r['side'])
    with np.load(os.path.join(staging, str(r['job']), 'gantry_ckpt_%d.npz' % r['job']), allow_pickle=True) as z:
        t_best = float(z['bestfit'])
        t_acc = json.loads(str(z['lbfgs_outcome']))['accepted']
    staged_as = 'copy'
    if os.path.exists(os.path.join(staging, 'meta.json')):
        with open(os.path.join(staging, 'meta.json')) as fh:
            for row in json.load(fh)['rows'].values():
                if row['job'] == r['job']:
                    staged_as = row['staged_as']
    acc = (r['lbfgs_outcome'] or {}).get('accepted')
    rel = abs(r['bestfit'] - t_best) / t_best
    checks = dict(accepted=acc == t_acc, bestfit_rel=bool(rel < 1e-3))
    probs = [k for k, ok in checks.items() if not ok]
    _finish(dict(verdict='FAIL' if probs else 'PASS', problems=probs, checks=checks, staged_as=staged_as,
                 bestfit=r['bestfit'], thesis_bestfit=t_best, rel=rel, accepted=acc, thesis_accepted=t_acc),
            path, label)


def floor_rms(csv_dir):
    """{(split, record): RMS per axis [m]} of the floor rows of res_errors.csv."""
    out = {}
    with open(os.path.join(csv_dir, 'res_errors.csv'), newline='') as fh:
        for row in csv.DictReader(fh):
            if row['model'] == 'floor':
                out[(row['split'], row['record'])] = np.array([float(row['rms_' + a]) for a in ('X1', 'X2', 'Y')])
    return out


def floor(a_out, b_out, csv_dir, path):
    ra, rb = _load(a_out), _load(b_out)
    label = 'floor %s %s' % (ra['phase'], ra['run'])
    probs = []
    if (ra['clean_tree'], rb['clean_tree']) != ('reverted', 'final') or ra['side'] != 'clean' or rb['side'] != 'clean':
        probs.append('needs clean outputs of the reverted (A) and final (B) trees, got %s %s / %s %s'
                     % (ra['side'], ra['clean_tree'], rb['side'], rb['clean_tree']))
    if (ra['phase'], ra['run'], ra['n_its']) != (rb['phase'], rb['run'], rb['n_its']):
        probs.append('different cases')
    if probs:
        _finish(dict(verdict='REFUSED', problems=probs), path, label)
    fl = floor_rms(csv_dir)
    pa, pb = np.load(os.path.join(a_out, 'preds.npz')), np.load(os.path.join(b_out, 'preds.npz'))
    rows, worst = [], 0.0
    for split in ('val', 'test'):
        recs = [str(n) for n in pa[split + '_records']]
        if recs != [str(n) for n in pb[split + '_records']]:
            probs.append('%s records differ' % split)
            continue
        d = np.sqrt(np.mean((pb[split + '_model'] - pa[split + '_model']) ** 2, axis=1))
        for n, dn in zip(recs, d):
            if (split, n) not in fl:
                probs.append('no floor row for %s %s' % (split, n))
                continue
            ratio = dn / fl[(split, n)]
            worst = max(worst, float(ratio.max()))
            rows.append(dict(split=split, record=n, diff_rms=dn.tolist(), floor_rms=fl[(split, n)].tolist(),
                             ratio=ratio.tolist()))
            if not np.all(ratio < 1):
                probs.append('%s %s ratio %s' % (split, n, np.round(ratio, 3).tolist()))
    _finish(dict(verdict='FAIL' if probs else 'PASS', problems=probs, worst_ratio=worst, rows=rows,
                 bestfit=(ra['bestfit'], rb['bestfit'])), path, label)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'bitwise' and len(sys.argv) == 5:
        bitwise(*sys.argv[2:])
    elif mode == 'thesis' and len(sys.argv) == 5:
        thesis(*sys.argv[2:])
    elif mode == 'floor' and len(sys.argv) == 6:
        floor(*sys.argv[2:])
    else:
        sys.exit(__doc__)
