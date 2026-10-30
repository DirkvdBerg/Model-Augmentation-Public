"""Bitwise comparison of two folders of black-box preds_BB*.npz, every array and key (D-285)."""
import sys
from pathlib import Path

import numpy as np

A, B = Path(sys.argv[1]), Path(sys.argv[2])
bad = 0
for fa in sorted(A.glob('preds_BB*.npz')):
    a, b = np.load(fa, allow_pickle=False), np.load(B / fa.name, allow_pickle=False)
    if set(a.files) != set(b.files):
        print('%s: keys %s vs %s' % (fa.name, sorted(a.files), sorted(b.files)))
        bad += 1
    for k in a.files:
        if k in b.files and (a[k].shape != b[k].shape or a[k].dtype != b[k].dtype or not np.array_equal(a[k], b[k])):
            print('%s[%s]: differs %s %s vs %s %s' % (fa.name, k, a[k].shape, a[k].dtype, b[k].shape, b[k].dtype))
            bad += 1
    print('%s: %d keys, %s records' % (fa.name, len(a.files), a['records'].shape[0]))
print('BITWISE IDENTICAL' if bad == 0 else '%d DIFFERENCES' % bad)
