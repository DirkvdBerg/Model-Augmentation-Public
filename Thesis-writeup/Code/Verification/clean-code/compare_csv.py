"""D-295: recomputed errors against the report's numbers, Results/24-10-2026/CSV/res_errors.csv.

    python -u compare_csv.py l4 L4_SIDE_DIR CSV_DIR REPORT.json     the merged l4_loader.py output of one side
    python -u compare_csv.py bb PREDS_DIR CSV_DIR REPORT.json       Clean/blackbox/eval_bb.py's preds_BB_<split>.npz

Every RMS and NRMS_e per split, run, model, record and axis present on both sides is compared. The CSV prints
'%.6e', seven significant digits, so a value matches when it lies within one unit of the 7th significant digit of
the printed value (D-294 amendment 1, data-derived). Also reported: how many print to the identical string, and
every CSV row of these runs and models that the recomputation lacks.
"""
import csv
import glob
import json
import math
import os
import sys

import numpy as np

AXES = ('X1', 'X2', 'Y')


def unit(c):
    """One unit of the 7th significant digit of the printed value c."""
    return 10.0 ** (math.floor(math.log10(abs(c))) - 6) if c != 0 else 0.0


def read_csv(csv_dir):
    out = {}
    with open(os.path.join(csv_dir, 'res_errors.csv'), newline='') as fh:
        for row in csv.DictReader(fh):
            out[(row['split'], row['run'], row['model'], row['record'])] = row
    return out


def recomputed_l4(side_dir):
    """{(split, run, model, record): (rms per axis, nrms_e per axis)} from l4_<run>_<split>.npz."""
    out = {}
    for p in sorted(glob.glob(os.path.join(side_dir, 'l4_*_*.npz'))):
        run, split = os.path.basename(p)[3:-4].rsplit('_', 1)
        with np.load(p) as z:
            for tag in [k[4:] for k in z.files if k.startswith('rms_')]:
                for n, a, b in zip(z['records'], z['rms_' + tag], z['nrmse_' + tag]):
                    out[(split, run, tag, str(n))] = (a, b)
    return out


def recomputed_bb(preds_dir):
    """As recomputed_l4, from the black-box prediction files (make_results.errors: run R8, models bb and floor)."""
    out = {}
    # eval_bb.py names the test file preds_BB.npz (make_results.sfx); the twins are no split of res_errors.csv
    for p in sorted(glob.glob(os.path.join(preds_dir, 'preds_BB*.npz'))):
        split = os.path.basename(p)[len('preds_BB'):-4].lstrip('_') or 'test'
        if split not in ('train', 'val', 'test'):
            continue
        with np.load(p) as z:
            for key, model in (('bb', 'bb'), ('y_nf', 'floor')):
                if key not in z.files:
                    continue
                for n, pr, y, sg in zip(z['records'], z[key], z['y'], z['sigma_e']):
                    a = np.sqrt(np.mean((pr - y) ** 2, axis=0))
                    out[(split, 'R8', model, str(n))] = (a, a / sg)
    return out


def compare(rec, ref):
    probs, n_val, n_same, worst = [], 0, 0, 0.0
    for key, (rms, nrms) in sorted(rec.items()):
        if key not in ref:
            probs.append('not in the CSV: %s' % (key,))
            continue
        row = ref[key]
        for vals, col in ((rms, 'rms_'), (nrms, 'nrms_e_')):
            for a, ax in zip(vals, AXES):
                c = float(row[col + ax])
                n_val += 1
                n_same += '%.6e' % a == row[col + ax]
                dev = abs(a - c) / unit(c) if c else abs(a)
                worst = max(worst, dev)
                if dev > 1:
                    probs.append('%s %s%s: %.9e vs CSV %s (%.2f units)' % (key, col, ax, a, row[col + ax], dev))
    covered = {k[:3] for k in rec}
    missing = sorted(k for k in ref if k[:3] in covered and k not in rec)
    probs += ['in the CSV, not recomputed: %s' % (k,) for k in missing]
    return dict(verdict='FAIL' if probs else ('PASS' if n_val else 'EMPTY'), values=n_val, identical_strings=n_same,
                worst_units=worst, problems=probs)


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] not in ('l4', 'bb'):
        sys.exit(__doc__)
    mode, src, csv_dir, path = sys.argv[1:]
    rec = recomputed_l4(src) if mode == 'l4' else recomputed_bb(src)
    report = dict(source=os.path.abspath(src), **compare(rec, read_csv(csv_dir)))
    with open(path, 'w') as fh:
        json.dump(report, fh, indent=1)
    print('COMPARE_CSV %s %s %s values=%d identical=%d worst=%.2f units problems=%d %s'
          % (mode, src, report['verdict'], report['values'], report['identical_strings'], report['worst_units'],
             len(report['problems']), json.dumps(report['problems'][:5])))
    sys.exit(0 if report['verdict'] == 'PASS' else 1)
