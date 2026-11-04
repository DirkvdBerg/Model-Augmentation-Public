"""D-295: the layer-4b outputs of one clean run against mainline's, bitwise.

    python -u compare_l4b.py MAINLINE_DIR CLEAN_DIR RUN [RUN ...]

Compares l4b_<run>.json (every number and string) and l4b_<run>.npz (every array) of each run; prints one
COMPARE_L4B line with PASS only when all are bitwise equal, else the largest relative difference per file.
"""
import json
import os
import sys

import numpy as np


def flat(o, p=''):
    if isinstance(o, dict):
        return sum((flat(v, '%s/%s' % (p, k)) for k, v in o.items()), [])
    if isinstance(o, (list, tuple)):
        return sum((flat(v, '%s[%d]' % (p, i)) for i, v in enumerate(o)), [])
    return [(p, o)]


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


if len(sys.argv) < 4:
    sys.exit(__doc__)
main_dir, clean_dir, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
probs = []
for r in runs:
    m = dict(flat(json.load(open(os.path.join(main_dir, 'l4b_%s.json' % r)))))
    c = dict(flat(json.load(open(os.path.join(clean_dir, 'l4b_%s.json' % r)))))
    if m.keys() != c.keys():
        probs.append('%s json keys differ' % r)
    d = {k: (rel(m[k], c[k]) if isinstance(m[k], float) else 1.0) for k in m if k in c and m[k] != c[k]}
    if d:
        w = max(d, key=d.get)
        probs.append('%s json: %d of %d differ, max rel %.1e at %s (%r vs %r)' % (r, len(d), len(m), d[w], w, m[w], c[w]))
    with np.load(os.path.join(main_dir, 'l4b_%s.npz' % r)) as zm, np.load(os.path.join(clean_dir, 'l4b_%s.npz' % r)) as zc:
        for k in zm.files:
            if k not in zc.files or zm[k].shape != zc[k].shape:
                probs.append('%s npz %s missing or shape differs' % (r, k))
            elif zm[k].tobytes() != zc[k].tobytes():
                probs.append('%s npz %s max rel %.1e' % (r, k, float(np.abs(zm[k] - zc[k]).max() / np.abs(zm[k]).max())))
print('COMPARE_L4B %s %s %s' % (' '.join(runs), 'FAIL' if probs else 'PASS', json.dumps(probs)))
sys.exit(1 if probs else 0)
