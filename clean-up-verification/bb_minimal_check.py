"""Equivalence of the minimal Clean/blackbox/ (D-285 amendment 7) with the thesis black box.

    python -u clean-up-verification/bb_minimal_check.py select
        Selection table of Clean/blackbox/select_bb.py on the 11 grid result.json files against
        Thesis-writeup/Results/24-10-2026/CSV/res_bb_selection.csv, every shared column as printed. Local, seconds.
    python -u clean-up-verification/bb_minimal_check.py preds R8_RUN_DIR OUT_DIR REF_DIR
        Clean/blackbox/eval_bb.py on the R8 run folder (index 9) into OUT_DIR, then every array of every
        preds_BB*.npz against REF_DIR (simulations/clean-golden/blackbox/eval_clean) bitwise. Server: the R8
        checkpoint is there.

The thesis checkpoints pickle the nets as jan_model.JanEncoder and jan_model.JanStateNet (the clean goldens as
bb.JanEncoder), deepSI's nets with .reshape for .view. Here only, both module names are aliases for deepSI's classes,
so a checkpoint loads into the nets the clean code uses.
"""
import csv
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'Clean'))

from deepSI.fit_systems.encoders import default_encoder_net, default_state_net  # noqa: E402

GRID = REPO / 'simulations' / 'clean-golden' / 'blackbox' / 'grid_89756'
REF_CSV = REPO / 'Thesis-writeup' / 'Results' / '24-10-2026' / 'CSV' / 'res_bb_selection.csv'
OUT_CSV = REPO / 'simulations' / 'clean-golden' / 'blackbox' / 'select_minimal' / 'selection.csv'


def check_select():
    from blackbox import select_bb
    select_bb.main(GRID, OUT_CSV)
    with open(REF_CSV, newline='') as fh:
        ref = list(csv.DictReader(fh))
    with open(OUT_CSV, newline='') as fh:
        new = list(csv.DictReader(fh))
    shared = [c for c in new[0] if c in ref[0]]
    bad = [(i, c, a[c], b[c]) for i, (a, b) in enumerate(zip(ref, new)) for c in shared if a[c] != b[c]]
    bad += [('rows', len(ref), len(new))] if len(ref) != len(new) else []
    print('shared columns:', ', '.join(shared))
    for d in bad:
        print('  differs:', d)
    print('SELECTION IDENTICAL' if not bad else '%d DIFFERENCES' % len(bad))
    return not bad


def check_preds(run, out, ref):
    for old in ('jan_model', 'bb'):
        alias = types.ModuleType(old)
        alias.JanEncoder, alias.JanStateNet = default_encoder_net, default_state_net
        sys.modules[old] = alias
    import torch
    from blackbox import eval_bb
    from blackbox.bb import N_THREADS
    torch.set_num_threads(N_THREADS)
    eval_bb.main(run, out)
    bad = 0
    files = sorted(ref.glob('preds_BB*.npz'))
    for fr in files:
        a, b = np.load(fr, allow_pickle=False), np.load(out / fr.name, allow_pickle=False)
        if set(a.files) != set(b.files):
            print('%s: keys %s vs %s' % (fr.name, sorted(a.files), sorted(b.files)))
            bad += 1
        for k in set(a.files) & set(b.files):
            if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype or not np.array_equal(a[k], b[k]):
                print('%s[%s] differs' % (fr.name, k))
                bad += 1
        print('%s: %d keys, %d records' % (fr.name, len(a.files), a['records'].shape[0]))
    ok = bad == 0 and len(files) == 4
    print('PREDICTIONS BITWISE IDENTICAL' if ok else '%d DIFFERENCES (%d files)' % (bad, len(files)))
    return ok


if __name__ == '__main__':
    if sys.argv[1] == 'select':
        ok = check_select()
    else:
        ok = check_preds(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    sys.exit(0 if ok else 1)
