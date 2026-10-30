"""Check of evaluate() against the training (D-284 amendment 9, E1): on a run folder evaluated on the CPU, the mean
validation NRMS_e of the augmented model in evaluation/scores.csv must equal the score the training stored for
the final checkpoint, exactly (same float).

The final checkpoint gantry_ckpt_<run_id> stores bestfit = the score of the history row 'final' in val_hist.npz,
which is a copy of the row whose weights the checkpoint holds: the accepted polish candidate, else the OBC model
with the basis rebuilt at the restored parameters, else the selected Adam checkpoint. The check names that row.

    python -u clean-up-verification/val_check.py RUN_DIR
"""
import csv
import os
import sys

import numpy as np

run_dir = sys.argv[1]
with open(os.path.join(run_dir, 'config.json')) as fh:
    import json
    run_id = str(json.load(fh)['run_id'])
stored = float(np.load(os.path.join(run_dir, 'gantry_ckpt_%s.npz' % run_id), allow_pickle=True)['bestfit'])
h = np.load(os.path.join(run_dir, 'val_hist.npz'))
events, scores = list(h['val_hist_event']), h['val_hist_score']
final = len(events) - 1 - events[::-1].index('final')
source = [i for i in range(final) if scores[i] == scores[final] and events[i] != 'selected']
print('stored score of the final checkpoint: %r (val_hist row final = %r; rows with that score: %s)'
      % (stored, float(scores[final]), [(str(events[i]), int(h['val_hist_update'][i])) for i in source]))

with open(os.path.join(run_dir, 'evaluation', 'scores.csv'), newline='') as fh:
    rows = [r for r in csv.DictReader(fh)
            if r['split'] == 'val' and r['frame'] == 'stage' and r['model'] == 'augmented']
records = list(dict.fromkeys(r['record'] for r in rows))
nrms = np.array([[float(r['nrms_e']) for r in rows if r['record'] == rec] for rec in records])
mine = float(np.mean(nrms))
ok = mine == stored
print('evaluate(): mean val NRMS_e of the augmented model %r over %d records x %d axes' % (mine, *nrms.shape))
print('VAL_CHECK %s (difference %.3e)' % ('PASS' if ok else 'FAIL', mine - stored))
sys.exit(0 if ok else 1)
