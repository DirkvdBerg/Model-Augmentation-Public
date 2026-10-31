"""evaluate() against the thesis evaluation (D-284 amendment 9): on the R1 final run folder (job 89792), the NRMS_e
and RMS per record and axis of the augmented model, the baseline with trained and with true parameters and the noise
floor must reproduce Thesis-writeup/Results/24-10-2026/CSV/res_errors.csv (rows run R1, models model, physics,
base_true; floor rows of the black-box cache) to the CSV's printed precision: each value printed with '%.6e' equals
the CSV entry.

Runs in the D-284 step-5 server job, which already stages the thesis checkpoints, once with the MM16 line of
Clean/gantry_dynamic/data.py reverted (the check) and once restored (the measured MM16 effect, never a tolerance):

    python -u clean-up-verification/r1_check.py RUN_DIR       (MM16 reverted: writes r1_check-reverted.json)
    python -u clean-up-verification/r1_check.py RUN_DIR       (MM16 restored: writes r1_check-final.json and
                                                               reports its largest difference to the reverted run)

RUN_DIR is the thesis folder; it is only read: convert_run.py converts it into r1_check-<state>-run/ (D-284
amendment 10, R1), which evaluate() reads and writes its evaluation/ into.
"""
import csv
import json
import os
import shutil
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'Clean'))

from gantry_dynamic.evaluation import evaluate                   # noqa: E402
from convert_run import convert_run                               # noqa: E402

CSV = os.path.join(REPO, 'Thesis-writeup', 'Results', '24-10-2026', 'CSV', 'res_errors.csv')
DATA_PY = os.path.join(REPO, 'Clean', 'gantry_dynamic', 'data.py')
# evaluate() model -> res_errors.csv model (the floor rows come from the black-box cache, run R8)
MODELS = {'augmented': ('R1', 'model'), 'trained': ('R1', 'physics'), 'true': ('R1', 'base_true')}
AXES = ('X1', 'X2', 'Y')

run_dir = sys.argv[1]
with open(DATA_PY, encoding='utf-8') as fh:
    state = ('reverted' if 'P_inv_T = np.linalg.inv(P.numpy().T).astype(cfg.dtype_np)' in fh.read() else 'final')
print('[r1_check] MM16 line: %s' % state, flush=True)
converted = os.path.join(HERE, 'r1_check-%s-run' % state)
shutil.rmtree(converted, ignore_errors=True)
print('[r1_check] converted: %s' % json.dumps(convert_run(run_dir, converted)['pairs']), flush=True)
res = evaluate(converted)

with open(CSV, newline='') as fh:
    ref = {(r['split'], r['run'], r['model'], r['record']): r for r in csv.DictReader(fh)}
mine, compared, bad = {}, 0, []
for split, frames in res.items():
    s = frames['stage']
    records = frames['records']
    for i, rec in enumerate(records):
        entries = [(key, MODELS[key], s['rms'][key][i], s['nrms'][key][i]) for key in MODELS]
        entries.append(('floor', ('R8', 'floor'), None, s['floor'][i]))
        for key, (run, model), rms, nrms in entries:
            row = ref.get((split, run, model, rec))
            mine['%s/%s/%s' % (split, rec, key)] = [float(v) for v in nrms]
            if row is None:
                bad.append('%s %s %s: not in res_errors.csv' % (split, rec, key))
                continue
            pairs = [('nrms_e_' + a, v) for a, v in zip(AXES, nrms)]
            pairs += [] if rms is None else [('rms_' + a, v) for a, v in zip(AXES, rms)]
            for col, v in pairs:
                compared += 1
                if '%.6e' % v != '%.6e' % float(row[col]):
                    bad.append('%s %s %s %s: evaluate %.6e, csv %s' % (split, rec, key, col, v, row[col]))
out = os.path.join(HERE, 'r1_check-%s.json' % state)
with open(out, 'w') as fh:
    json.dump(dict(state=state, run_dir=run_dir, compared=compared, mismatches=bad, nrms=mine), fh, indent=1)
print('[r1_check] %d values compared, %d differ at the printed precision' % (compared, len(bad)))
for b in bad[:20]:
    print('    ' + b)
other = os.path.join(HERE, 'r1_check-reverted.json')
if state == 'final' and os.path.exists(other):
    with open(other) as fh:
        prev = json.load(fh)['nrms']
    d = max(abs(a - b) / abs(b) for k in mine if k in prev for a, b in zip(mine[k], prev[k]) if b)
    print('[r1_check] MM16 effect: largest relative NRMS_e difference to the reverted run %.3e' % d)
print('R1_CHECK %s (%s)' % ('PASS' if not bad else 'FAIL', state))
sys.exit(0 if not bad or state == 'final' else 1)
