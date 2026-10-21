"""Compare the layer-4/4b loader outputs of the two code roots, bitwise (design 6.3 rows 4 and 4b).

    python -u compare_l4.py MAINLINE_OUT CLEAN_OUT

REFUSED, nothing compared, unless the clean summary.json records the clean source manifest of the
current Clean/ tree (H2), and on either side when a loader output present differs from the SHA256
its summary.json lists (or is not listed: written by another loader run). Then both sides must hold the full expected set (H1): the nine final runs,
the three layer-4 splits of each, every layer-4b array and row l4_loader.l4b writes per run, a
val-score check per run on both sides with a finite, positive ref_score, and the mainline evaluation
check over every run and part. Anything missing is PARTIAL (listed), never PASS; a value mismatch is
FAIL. Also reports each side's val-score acceptance (finite, positive, within 1e-3 of the recorded
bestfit) and the mainline loader's agreement with the unmodified evaluation (--eval-check).
"""
import glob
import json
import os
import sys

import numpy as np

import hcommon as H

REPORT = os.path.join(H.HERE, 'compare-l4.json')
# the expected set, from l4_loader.py: the default --runs, STACKS, BASE_ROWS, and what l4() and l4b() write
RUNS = ('R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R13', 'R14')
SPLITS = ('train', 'val', 'test')
BASE_ROWS = ('R1', 'R4', 'R6', 'R13')
L4B_ARRAYS = {'R1': {'deployed_basis_R1_train'}, 'R4': {'frozen_y_R4_val'}}
L4B_ROWS = {'R1': {'overlap', 'bias', 'negation'}, 'R4': {'overlap', 'bias', 'negation', 'added_poles'},
            'R7': {'added_poles'}, 'R13': {'overlap', 'bias'}}


def diff(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return 'dtype/shape %s %s vs %s %s' % (a.dtype, a.shape, b.dtype, b.shape)
    if a.tobytes() == b.tobytes():
        return None
    if a.dtype.kind == 'f':
        d = float(np.nanmax(np.abs(a - b)))
        return 'max abs %.3e (rel %.3e)' % (d, d / max(float(np.nanmax(np.abs(a))), 1e-300))
    return '%d entries differ' % int((a != b).sum())


def expected_files():
    return (['l4_%s_%s.npz' % (r, s) for r in RUNS for s in SPLITS]
            + ['l4b_%s.%s' % (r, e) for r in RUNS for e in ('npz', 'json')])


def l4_keys(run, split):
    """The arrays l4_loader.l4 writes for `run` on `split`."""
    tags = ('model',) + (('physics', 'base_start', 'base_true') if run in BASE_ROWS else ())
    keys = {'records', 'k0', 'y', 'sigma_e', 'params_learned', 'params_init'}
    keys |= {p + t for t in tags for p in ('', 'rms_', 'nrmse_')}
    return keys | ({'val_score', 'ref_score'} if split == 'val' else set())


def eval_keys(run):
    """The items l4_loader.eval_check compares for `run`, both parts."""
    tags = ('model',) + (('physics', 'base_start', 'base_true') if run in BASE_ROWS else ())
    keys = {'%s/%s' % (s, k) for s in SPLITS for k in ('y', 'sigma_e', 'params_learned') + tags}
    return keys | {'val/val_score', 'val/ref_score'} | L4B_ROWS.get(run, set())


def row_gaps(run, rows):
    """What one l4b_<run>.json lacks of the rows l4_loader.l4b writes for `run`."""
    gaps = []
    for k in sorted(L4B_ROWS.get(run, ())):
        r = rows.get(k)
        if not r:
            gaps.append(k)
        elif k == 'added_poles':
            if len(r) != 4:
                gaps.append('added_poles: %d of 4 values' % len(r))
        else:
            got = ({x['stack'] for x in r} if k == 'overlap' else
                   {x[1] for x in r if k == 'bias' or x[2] == 'pooled'})
            if got != set(SPLITS):
                gaps.append('%s: stacks %s' % (k, sorted(got)))
    return gaps


def npz_keys(p):
    with np.load(p) as z:
        return set(z.files)


def side_gaps(d, side):
    """H1: what the expected array keys and rows lack in the files present on one side."""
    gaps = []
    for run in RUNS:
        for split in SPLITS:
            p = os.path.join(d, 'l4_%s_%s.npz' % (run, split))
            miss = l4_keys(run, split) - npz_keys(p) if os.path.exists(p) else set()
            if miss:
                gaps.append('%s: l4_%s_%s.npz lacks %s' % (side, run, split, sorted(miss)))
        p = os.path.join(d, 'l4b_%s.npz' % run)
        miss = L4B_ARRAYS.get(run, set()) - npz_keys(p) if os.path.exists(p) else set()
        if miss:
            gaps.append('%s: l4b_%s.npz lacks %s' % (side, run, sorted(miss)))
        p = os.path.join(d, 'l4b_%s.json' % run)
        miss = row_gaps(run, json.load(open(p))) if os.path.exists(p) else []
        if miss:
            gaps.append('%s: l4b_%s.json lacks %s' % (side, run, miss))
    return gaps


def stale(d, side):
    """[] when every loader output present in d is listed in its summary.json `outputs` with the
    SHA256 it has now, else the problems (a file the last loader run did not write, or rewrote after)."""
    p = os.path.join(d, 'summary.json')
    listed = (json.load(open(p)) if os.path.exists(p) else {}).get('outputs')
    if listed is None:
        return ['%s: summary.json lists no outputs' % side]
    probs = []
    for name in sorted(set(expected_files()) | set(listed)):
        f = os.path.join(d, name)
        if not os.path.exists(f):
            if name in listed:
                probs.append('%s: %s listed in summary.json but missing' % (side, name))
        elif name not in listed:
            probs.append('%s: %s not written by the run that wrote summary.json' % (side, name))
        elif H.sha256_file(f) != listed[name]:
            probs.append('%s: %s differs from its summary.json hash' % (side, name))
    return probs


def refusal(cdir, mdir):
    """H2: [] when the clean output was made from the current Clean/ tree and neither side holds a
    stale output, else the problems."""
    p = os.path.join(cdir, 'summary.json')
    s = json.load(open(p)) if os.path.exists(p) else {}
    probs = [] if s.get('side') == 'clean' else ['%s does not record side clean' % p]
    return probs + H.clean_tree_problems(s.get('clean_tree')) + stale(mdir, 'mainline') + stale(cdir, 'clean')


def main(mdir, cdir):
    probs = refusal(cdir, mdir)
    if probs:
        for x in probs:
            print('REFUSED: %s' % x)
        with open(REPORT, 'w') as fh:
            json.dump({'status': 'REFUSED', 'problems': probs}, fh, indent=1)
        print('COMPARE_L4 REFUSED')
        return 'REFUSED'
    res, failed, missing = {}, [], []
    found = {os.path.basename(p) for d in (mdir, cdir) for pat in ('l4*.npz', 'l4b_*.json')
             for p in glob.glob(os.path.join(d, pat))}
    for name in sorted(set(expected_files()) | found):
        p, q = os.path.join(mdir, name), os.path.join(cdir, name)
        absent = [s for s, f in (('mainline', p), ('clean', q)) if not os.path.exists(f)]
        if absent:
            res[name] = 'MISSING on the %s side' % ' and '.join(absent)
            missing.append('%s (%s)' % (name, ' and '.join(absent)))
            continue
        if name.endswith('.json'):
            same = json.load(open(p)) == json.load(open(q))
            res[name] = 'PASS' if same else 'FAIL (rows differ)'
            if not same:
                failed.append(name)
            continue
        a, b = np.load(p, allow_pickle=True), np.load(q, allow_pickle=True)
        bad = {}
        if set(a.files) != set(b.files):
            bad['keys'] = '%s vs %s' % (sorted(a.files), sorted(b.files))
        for k in sorted(set(a.files) & set(b.files)):
            d = diff(np.asarray(a[k]), np.asarray(b[k]))
            if d:
                bad[k] = d
        res[name] = bad or 'PASS'
        if bad:
            failed.append(name)
    missing += side_gaps(mdir, 'mainline') + side_gaps(cdir, 'clean')
    for side, d in (('mainline', mdir), ('clean', cdir)):
        p = os.path.join(d, 'summary.json')
        if not os.path.exists(p):
            missing.append('%s: summary.json' % side)
            continue
        s = json.load(open(p))
        checks = s.get('checks', {})
        for run in sorted(set(RUNS) | set(checks)):
            v = checks.get(run, {}).get('val_score')
            if v is None:
                missing.append('%s: val_score check of %s' % (side, run))
                continue
            res['%s:%s:val_score' % (side, run)] = v
            r = v.get('ref_score')
            if not (isinstance(r, (int, float)) and np.isfinite(r) and r > 0):
                missing.append('%s: finite, positive ref_score in the val_score check of %s' % (side, run))
            if not v.get('ok'):
                failed.append('%s:%s:val_score' % (side, run))
        if side == 'mainline':
            ev = s.get('eval_check', {})
            res['mainline:eval_check'] = ev
            for run in RUNS:
                miss = eval_keys(run) - set(ev.get(run, {}))
                if miss:
                    missing.append('mainline: eval_check of %s lacks %s' % (run, sorted(miss)))
            failed += ['mainline:eval_check:%s:%s' % (run, k) for run, c in ev.items()
                       for k, v in c.items() if not v]
    status = 'FAIL' if failed else 'PARTIAL' if missing else 'PASS'
    for k, v in res.items():
        print('%-34s %s' % (k, v))
    for x in missing:
        print('MISSING: %s' % x)
    res['missing'], res['failed'], res['status'] = missing, failed, status
    with open(REPORT, 'w') as fh:
        json.dump(res, fh, indent=1, default=str)
    print('COMPARE_L4 %s' % status)
    return status


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) == 'PASS' else 1)
