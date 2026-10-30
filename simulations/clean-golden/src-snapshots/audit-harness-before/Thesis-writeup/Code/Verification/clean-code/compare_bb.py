"""Bitwise comparison of two black-box runs (D-285): result.json, val_hist.json, _best.pth, _last.pth.
Run from the repo root: python Thesis-writeup/Code/Verification/clean-code/compare_bb.py <golden dir> <other dir>"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

for d in ('scripts/gantry/blackbox-cl-trainable/bbcl', 'Clean/blackbox'):   # pickled JanEncoder etc.
    sys.path.insert(0, str(Path(d).resolve()))
A, B = Path(sys.argv[1]), Path(sys.argv[2])
VOLATILE = {'wall_s', 'out', 'name', 'checkpoint', 'time', 'start_t', 'total_time', 'time_val', 'time_loss',
            'time_back', 'train_time', 'checkpoint_dir'}
diffs, notes = [], []


def cmp(a, b, path):
    key = path.rsplit('.', 1)[-1]
    if key in VOLATILE:
        return
    if isinstance(a, torch.nn.Module):
        sa, sb = a.state_dict(), b.state_dict()
        if sa.keys() != sb.keys():
            diffs.append('%s: module keys differ' % path)
        for k in sa:
            if not torch.equal(sa[k], sb[k]):
                diffs.append('%s.%s: max abs diff %.3e' % (path, k, (sa[k] - sb[k]).abs().max()))
    elif isinstance(a, torch.optim.Optimizer):
        cmp(a.state_dict(), b.state_dict(), path)
    elif isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            diffs.append('%s: tensor differs' % path)
    elif isinstance(a, dict):
        if set(a) != set(b):
            notes.append('%s: only in A %s, only in B %s' % (path, sorted(set(a) - set(b)), sorted(set(b) - set(a))))
        for k in a:
            if k in b:
                cmp(a[k], b[k], '%s.%s' % (path, k))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diffs.append('%s: length %d vs %d' % (path, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            cmp(x, y, '%s[%d]' % (path, i))
    elif isinstance(a, np.ndarray):
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
            diffs.append('%s: array differs' % path)
    elif isinstance(a, float):
        if not (a == b or (np.isnan(a) and np.isnan(b))):
            diffs.append('%s: %r vs %r' % (path, a, b))
    elif isinstance(a, (int, str, bool, type(None), np.generic)):
        if a != b:
            diffs.append('%s: %r vs %r' % (path, a, b))


for f in ('result.json', 'val_hist.json'):
    cmp(json.load(open(A / f)), json.load(open(B / f)), f)
ck = 'localappdata/deepSI/checkpoints'
for suffix in ('_best.pth', '_last.pth'):
    fa, fb = sorted((A / ck).glob('*' + suffix)), sorted((B / ck).glob('*' + suffix))
    if len(fa) != 1 or len(fb) != 1:
        diffs.append('%s: %d and %d files' % (suffix, len(fa), len(fb)))
        continue
    cmp(torch.load(fa[0], map_location='cpu', weights_only=False),
        torch.load(fb[0], map_location='cpu', weights_only=False), fa[0].name.split('_')[-1])
    print('compared %s and %s' % (fa[0].name, fb[0].name))
print('\n'.join('note: ' + n for n in notes))
print('\n'.join(diffs) if diffs else 'BITWISE IDENTICAL (volatile keys skipped: %s)' % sorted(VOLATILE))
