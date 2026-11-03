"""D-294 arm A: swap the float64 theta^0 and the exact P^-T of Clean/ back to the float32 forms of the thesis runs.

    python -u d294_revert.py apply                          final -> reverted (refused unless the tree is final)
    python -u d294_revert.py restore                        reverted -> final (refused unless the tree is reverted)
    python -u d294_revert.py status [--expect final|reverted]

Four whole-line swaps in two files, each line matched exactly once, so the line numbers (coverage) stay the same:
theta^0 and P in float32 and P_INV_T the float64 inverse of the float32 P (gantry_ss.py; Clean/ before D-294,
the inverse of mainline model.py and obc_gantry.py); P imported into data.py, whose compute_normalization takes q
from the float32 inverse, as mainline data.py (MM16 reverted). The reverted tree is what the bitwise comparators
prove against mainline; compare_d294.py proves that final and reverted differ in these lines only. Prints the
state and the SHA256 of both files; a tree that is neither final nor reverted is 'mixed' and refused.
"""
import argparse
import hashlib
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
CLEAN = os.path.join(REPO, 'Clean')
# (file below Clean/, line in the final tree, line in the reverted tree)
SWAPS = (
    ('model_augmentation/systems/gantry_ss.py',
     '_D = torch.float64',
     '_D = torch.float32'),
    ('model_augmentation/systems/gantry_ss.py',
     'P_INV_T = np.array([[0.5, 0.5, 0.0], [1 / float(Lb), -1 / float(Lb), 0.0], [0.0, 0.0, 1.0]])',
     'P_INV_T = np.linalg.inv(P.detach().cpu().numpy().astype(np.float64).T)'),
    ('gantry_dynamic/data.py',
     'from model_augmentation.systems.gantry_ss import Cd, Dd, P_INV_T',
     'from model_augmentation.systems.gantry_ss import Cd, Dd, P_INV_T, P'),
    ('gantry_dynamic/data.py',
     '        q = record.y @ P_INV_T.T',
     '        q = (np.linalg.inv(P.numpy().T).astype(cfg.dtype_np) @ record.y.T).T'),
)
FILES = sorted({f for f, _, _ in SWAPS})


def _read(rel):
    with open(os.path.join(CLEAN, *rel.split('/')), 'rb') as fh:
        return fh.read()


def _count(lines, text):
    return sum(1 for ln in lines if ln.rstrip(b'\r\n') == text.encode())


def state(files=None):
    """'final', 'reverted' or 'mixed' for the file contents `files` (rel -> bytes; default: the files on disk)."""
    files = files or {f: _read(f) for f in FILES}
    counts = [(_count(files[f].splitlines(True), a), _count(files[f].splitlines(True), b)) for f, a, b in SWAPS]
    if all(c == (1, 0) for c in counts):
        return 'final'
    if all(c == (0, 1) for c in counts):
        return 'reverted'
    return 'mixed'


def swapped(files, to):
    """`files` (rel -> bytes) with every swap applied towards `to` ('reverted' or 'final'), line endings kept."""
    out = dict(files)
    for f, a, b in SWAPS:
        old, new = (a, b) if to == 'reverted' else (b, a)
        lines = out[f].splitlines(True)
        hit = [i for i, ln in enumerate(lines) if ln.rstrip(b'\r\n') == old.encode()]
        if len(hit) != 1:
            raise SystemExit('%s: %d lines match %r, expected 1' % (f, len(hit), old))
        i = hit[0]
        lines[i] = new.encode() + lines[i][len(lines[i].rstrip(b'\r\n')):]
        out[f] = b''.join(lines)
    return out


def reverted_hashes():
    """SHA256 of each swapped file as the reverted tree holds it, computed from the files on disk in either state."""
    files = {f: _read(f) for f in FILES}
    if state(files) == 'final':
        files = swapped(files, 'reverted')
    return {f: hashlib.sha256(files[f]).hexdigest() for f in FILES}


def report():
    s = state()
    print('D294_STATE %s' % s)
    for f in FILES:
        print('  %s  %s' % (hashlib.sha256(_read(f)).hexdigest(), f))
    return s


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('apply', 'restore', 'status'))
    ap.add_argument('--expect', choices=('final', 'reverted'))
    args = ap.parse_args()
    if args.action == 'status':
        s = report()
        sys.exit(1 if s == 'mixed' or (args.expect and s != args.expect) else 0)
    need, to = ('final', 'reverted') if args.action == 'apply' else ('reverted', 'final')
    files = {f: _read(f) for f in FILES}
    if state(files) != need:
        sys.exit('the tree is %s, %s needs %s' % (state(files), args.action, need))
    for f, b in swapped(files, to).items():
        with open(os.path.join(CLEAN, *f.split('/')), 'wb') as fh:
            fh.write(b)
    sys.exit(0 if report() == to else 1)
