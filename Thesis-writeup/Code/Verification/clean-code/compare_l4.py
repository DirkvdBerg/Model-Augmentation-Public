"""Compare the layer-4/4b loader outputs of the two code roots, bitwise (design 6.3 rows 4 and 4b).

    python -u compare_l4.py MAINLINE_OUT CLEAN_OUT

Also reports each side's val-score acceptance (finite, positive, within 1e-3 of the recorded bestfit)
and the mainline loader's agreement with the unmodified evaluation (--eval-check).
"""
import glob
import json
import os
import sys

import numpy as np


def diff(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return 'dtype/shape %s %s vs %s %s' % (a.dtype, a.shape, b.dtype, b.shape)
    if a.tobytes() == b.tobytes():
        return None
    if a.dtype.kind == 'f':
        d = float(np.nanmax(np.abs(a - b)))
        return 'max abs %.3e (rel %.3e)' % (d, d / max(float(np.nanmax(np.abs(a))), 1e-300))
    return '%d entries differ' % int((a != b).sum())


def main(mdir, cdir):
    res, ok = {}, True
    for p in sorted(glob.glob(os.path.join(mdir, 'l4*.npz'))):
        name = os.path.basename(p)
        q = os.path.join(cdir, name)
        if not os.path.exists(q):
            res[name] = 'MISSING on the clean side'
            ok = False
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
        ok &= not bad
    for p in sorted(glob.glob(os.path.join(mdir, 'l4b_*.json'))):
        name = os.path.basename(p)
        q = os.path.join(cdir, name)
        same = os.path.exists(q) and json.load(open(p)) == json.load(open(q))
        res[name] = 'PASS' if same else 'FAIL (rows differ or missing)'
        ok &= same
    for side, d in (('mainline', mdir), ('clean', cdir)):
        s = json.load(open(os.path.join(d, 'summary.json')))
        for run, c in s['checks'].items():
            v = c.get('val_score')
            if v is not None:
                res['%s:%s:val_score' % (side, run)] = v
                ok &= bool(v['ok'])
        if side == 'mainline':
            res['mainline:eval_check'] = s['eval_check']
            ok &= all(all(v.values()) for v in s['eval_check'].values())
    for k, v in res.items():
        print('%-34s %s' % (k, v))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compare-l4.json'), 'w') as fh:
        json.dump(res, fh, indent=1, default=str)
    print('COMPARE_L4 %s' % ('PASS' if ok else 'FAIL'))
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
