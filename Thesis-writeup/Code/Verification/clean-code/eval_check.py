"""D-295: Clean/'s own evaluation (gantry_dynamic/evaluation.py) of a thesis run, against the report.

    python -u eval_check.py STAGED_RUN_DIR OUT CSV_DIR RUN

Converts the staged thesis folder into OUT/run (clean-up-verification/convert_run.py), runs evaluate(OUT/run), which
writes OUT/run/evaluation/ (figures, scores.csv, params.csv, predictions.npz), and compares every stage-frame RMS and
NRMS_e of scores.csv with res_errors.csv of RUN: models augmented, trained, true are the report's model, physics,
base_true; a value matches within one unit of the printed 7th significant digit (as compare_csv.py).
"""
import csv
import os
import sys

import hcommon as H

if len(sys.argv) != 5:
    sys.exit(__doc__)
staged, out, csv_dir, run = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3], sys.argv[4]
H.select_side('clean')
sys.path.append(os.path.join(H.REPO, 'clean-up-verification'))
from convert_run import convert_run  # noqa: E402
from gantry_dynamic.evaluation import evaluate  # noqa: E402
from compare_csv import unit  # noqa: E402

run_dir = os.path.join(out, 'run')
convert_run(staged, run_dir)
evaluate(run_dir)
H.assert_roots()

MODEL = {'augmented': 'model', 'trained': 'physics', 'true': 'base_true'}
ref = {}
with open(os.path.join(csv_dir, 'res_errors.csv'), newline='') as fh:
    for row in csv.DictReader(fh):
        if row['run'] == run:
            ref[(row['split'], row['model'], row['record'])] = row
probs, n, same, worst = [], 0, 0, 0.0
with open(os.path.join(run_dir, 'evaluation', 'scores.csv'), newline='') as fh:
    for row in csv.DictReader(fh):
        if row['frame'] != 'stage' or row['model'] not in MODEL:
            continue
        key = (row['split'], MODEL[row['model']], row['record'])
        if key not in ref:
            probs.append('not in the CSV: %s' % (key,))
            continue
        for col, ccol in (('rms', 'rms_'), ('nrms_e', 'nrms_e_')):
            a, c = float(row[col]), ref[key][ccol + row['axis']]
            n += 1
            same += '%.6e' % a == c
            dev = abs(a - float(c)) / unit(float(c)) if float(c) else abs(a)
            worst = max(worst, dev)
            if dev > 1:
                probs.append('%s %s %s: %.9e vs CSV %s (%.2f units)' % (key, col, row['axis'], a, c, dev))
verdict = 'FAIL' if probs or not n else 'PASS'
print('EVAL_CHECK %s %s values=%d identical=%d worst=%.2f units problems=%d %s'
      % (run, verdict, n, same, worst, len(probs), probs[:5]), flush=True)
sys.exit(0 if verdict == 'PASS' else 1)
