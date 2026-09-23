"""Telica LPV dataset index (split by operating point), from docs/kamtin-telica-schema.md.

Paths are built from the schema, never by listing the blocked folder. Only the vendored loader
opens the files. `xpos`/`ypos` are the start position in mm; each run moves 40 mm in X and 80 mm
in Y from there.
"""
import os
import re

import tr_env

DATA = os.path.join(tr_env.REPO, 'kamtin-data', 'Data Telica', '06 40 mm XL 80 mm YL')

SPLIT = {
    'train': ['xpos_-60_ypos-40', 'xpos_-60_ypos120', 'xpos_-60_ypos-120', 'xpos_-60_ypos-200',
              'xpos_-135_ypos40', 'xpos_-135_ypos-40', 'xpos_-135_ypos-200',
              'xpos_-210_ypos40', 'xpos_-210_ypos120', 'xpos_-210_ypos-120',
              'xpos_-210_ypos-200'],
    'validation': ['xpos_-135_ypos120', 'xpos_-210_ypos-40'],
    'test': ['xpos_-135_ypos-120', 'xpos_-60_ypos40'],
}
# iter0 = feedforward off (exact controller I/O pair); iterETEL = ETEL feedforward;
# iter1..8 ILC (iter9/10 in four train folders, iterTEST in test folders).
ITERS_COMMON = ['iter0'] + [f'iter{i}' for i in range(1, 9)] + ['iterETEL']
ITERS_OPTIONAL = ['iter9', 'iter10', 'iterTEST']


def op_xy_mm(op):
    m = re.match(r'xpos_(-?\d+)_ypos(-?\d+)', op)
    return float(m.group(1)), float(m.group(2))


def path(split, op, it='iter0'):
    return os.path.join(DATA, split, op, it + '.log')


def records(splits=('train', 'validation', 'test'), iters=('iter0',), optional=False):
    """[(split, op, iter, path)] for existing files. `optional` adds iter9/10/TEST where present."""
    out = []
    for s in splits:
        for op in SPLIT[s]:
            its = list(iters) + (ITERS_OPTIONAL if optional else [])
            for it in its:
                p = path(s, op, it)
                if os.path.isfile(p):
                    out.append((s, op, it, p))
    return out
