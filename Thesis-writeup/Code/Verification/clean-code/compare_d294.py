"""D-294 arm B: the final Clean/ tree against arm A (the same tree with d294_revert.py applied), both clean.

    python -u compare_d294.py l12 A_ROOT B_ROOT [221-lowpass ...]   clean layer-1/2 captures (HARNESS_CLEAN_OUT roots)
    python -u compare_d294.py l4 A_DIR B_DIR                         clean l4_loader.py outputs

REFUSED, nothing compared, unless the tree is final now, B was made from it (H2) and A from the same tree with
only the d294_revert.py lines swapped (its two files hash as the reverted ones, every other file as now), and
neither side holds a stale output. Then:
  l12  arrays that depend on neither theta^0 nor q (INDEPENDENT) and every info entry must be bitwise equal; the
       others (normalisation of x, OBC tuples, encoder start, forward, loss, gradients) are reported, not judged.
  l4   records, k0, y, sigma_e, ref_score and the parameters (loaded from the checkpoint) must be bitwise equal;
       every thesis number (RMS, NRMS_e, val_score; overlap, bias, negation and pole rows) must differ from arm A
       by less than one unit in its 7th significant digit, the last digit the result CSVs print ('%.6e',
       Thesis-writeup/Results/24-10-2026/CSV); the predicted trajectories and l4b arrays are reported.
Writes compare-d294-<layer>.json next to this script.
"""
import json
import os
import sys

import numpy as np

import hcommon as H
import d294_revert
from compare import L12_CASES, check_fingerprint, diff_arrays
from compare_l4 import expected_files, stale

# layer-1 arrays computed before theta^0 or q enters: data, u and y normalisation, controller bank and rows, loss
# weights, training windows and batch (normalised u and y only), the encoder's random start, the regressor windows
# of the least-squares start and their singular values
INDEPENDENT_PREFIX = ('data/', 'fsnorm/', 'bank/', 'sim/', 'windows/', 'batch/', 'build/encoder/')
INDEPENDENT = {'norm/std_u', 'norm/u_mean', 'norm/ystd', 'norm/y0', 'ls/uh', 'ls/yh', 'ls/rank',
               'ls/singular_values'}
L4_BITWISE = {'records', 'k0', 'y', 'sigma_e', 'ref_score', 'params_learned', 'params_init'}


def independent(k):
    return k in INDEPENDENT or k.startswith(INDEPENDENT_PREFIX)


def tree_problems(a_tree, b_tree):
    """[] when the tree is final, B is the current tree and A differs from it only by the reverted D-294 files."""
    probs = [] if d294_revert.state() == 'final' else ['the tree is %s, not final' % d294_revert.state()]
    probs += ['B: ' + p for p in H.clean_tree_problems(b_tree)]
    if not a_tree:
        return probs + ['A records no clean source manifest']
    want = dict(H.clean_tree_hashes(), **d294_revert.reverted_hashes())
    probs += ['A: %s is not the current file%s' % (k, ' reverted' if k in d294_revert.FILES else '')
              for k in sorted(set(want) | set(a_tree)) if want.get(k) != a_tree.get(k)]
    return probs


def printed(a, b):
    """(ok, worst): |a - b| below one unit in the 7th significant digit of a ('%.6e'); worst = max |a - b| / unit."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return False, float('inf')
    d = np.abs(a - b)
    with np.errstate(divide='ignore', invalid='ignore'):
        unit = 10.0 ** (np.floor(np.log10(np.abs(a))) - 6)
        ratio = np.where(a == b, 0.0, np.where(a != 0, d / unit, np.inf))
    ratio = np.where(np.isnan(a) & np.isnan(b), 0.0, ratio)
    ratio = np.where(np.isnan(ratio), np.inf, ratio)
    worst = float(ratio.max()) if ratio.size else 0.0
    return worst < 1.0, worst


def _reported(out, k, a, b):
    d = diff_arrays(np.asarray(a), np.asarray(b))
    if d:
        out['reported'][k] = d


def compare_l1l2(adir, bdir, stem, out):
    a, b = np.load(os.path.join(adir, stem + '.npz')), np.load(os.path.join(bdir, stem + '.npz'))
    aj, bj = (json.load(open(os.path.join(d, stem + '.json'))) for d in (adir, bdir))
    if set(a.files) != set(b.files) or set(aj['hashed']) != set(bj['hashed']):
        out['failed']['%s:keys' % stem] = 'A %s | B %s' % (sorted(a.files), sorted(b.files))
    for k in sorted(set(a.files) & set(b.files)):
        if stem == 'L1' and independent(k):
            d = diff_arrays(a[k], b[k])
            if d:
                out['failed'][k] = d
        else:
            _reported(out, '%s/%s' % (stem, k), a[k], b[k])
    for k in sorted(set(aj['hashed']) & set(bj['hashed'])):
        if aj['hashed'][k] != bj['hashed'][k]:
            target = out['failed'] if stem == 'L1' and independent(k) else out['reported']
            target['%s/%s' % (stem, k)] = 'hash differs (A %s | B %s)' % (aj['hashed'][k], bj['hashed'][k])
    for k in sorted(set(aj['info']) | set(bj['info'])):
        if aj['info'].get(k) != bj['info'].get(k):
            out['failed']['%s:info:%s' % (stem, k)] = 'A %s | B %s' % (str(aj['info'].get(k))[:200],
                                                                      str(bj['info'].get(k))[:200])


def run_l12(aroot, broot, cases):
    res, ok_all = {}, True
    for case in cases or L12_CASES:
        dirs = {s: (os.path.join(aroot, s, case), os.path.join(broot, s, case)) for s in ('L1', 'L2')}
        metas = {s: [json.load(open(os.path.join(d, 'meta.json'))) if os.path.exists(os.path.join(d, 'meta.json'))
                     else None for d in dirs[s]] for s in dirs}
        if any(m is None for s in metas for m in metas[s]):
            res[case] = {'status': 'NO OUTPUT'}
            ok_all = False
            print('%-14s NO OUTPUT' % case, flush=True)
            continue
        probs = []
        for s, (am, bm) in metas.items():
            probs += ['%s: %s' % (s, p) for p in tree_problems(am['sources'].get('clean'), bm['sources'].get('clean'))]
            # check_fingerprint: mainline sources, data, files, libraries, budget, threads; its H2 line is B's
            probs += ['%s: %s' % (s, p) for p in check_fingerprint(am, bm)]
            probs += ['%s A: %s' % (s, p) for p in H.output_problems(dirs[s][0], am)]
            probs += ['%s B: %s' % (s, p) for p in H.output_problems(dirs[s][1], bm)]
        if probs:
            res[case] = {'status': 'REFUSED', 'problems': sorted(set(probs))}
            ok_all = False
            print('%-14s REFUSED %s' % (case, res[case]['problems']), flush=True)
            continue
        out = {'failed': {}, 'reported': {}}
        for s in ('L1', 'L2'):
            compare_l1l2(dirs[s][0], dirs[s][1], s, out)
        ok = not out['failed']
        ok_all &= ok
        res[case] = dict(status='PASS' if ok else 'FAIL', **out)
        print('%-14s %s  (%d independent differences, %d reported)' % (case, res[case]['status'],
                                                                         len(out['failed']), len(out['reported'])))
        for k, v in list(out['failed'].items())[:12] + list(out['reported'].items())[:12]:
            print('    %s: %s' % (k, v))
    return res, ok_all


def _rows(a, b, path, out):
    """l4b rows: strings, ints and bools equal, floats within the printed precision."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out['failed'][path + ':keys'] = 'A %s | B %s' % (sorted(a), sorted(b))
        for k in sorted(set(a) & set(b)):
            _rows(a[k], b[k], '%s/%s' % (path, k), out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out['failed'][path + ':len'] = '%d vs %d' % (len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            _rows(x, y, '%s/%d' % (path, i), out)
    elif isinstance(a, float) or isinstance(b, float):
        ok, worst = printed(a, b)
        out['printed_worst'] = max(out['printed_worst'], worst)
        if not ok:
            out['failed'][path] = 'A %r | B %r (%.2f units of the 7th digit)' % (a, b, worst)
    elif a != b:
        out['failed'][path] = 'A %r | B %r' % (a, b)


def run_l4(adir, bdir):
    sa, sb = (json.load(open(os.path.join(d, 'summary.json'))) if os.path.exists(os.path.join(d, 'summary.json'))
              else {} for d in (adir, bdir))
    probs = ['%s does not record side clean' % s for s, m in (('A', sa), ('B', sb)) if m.get('side') != 'clean']
    probs += tree_problems(sa.get('clean_tree'), sb.get('clean_tree')) + stale(adir, 'A') + stale(bdir, 'B')
    if sa.get('staging') != sb.get('staging') or sa.get('runs') != sb.get('runs'):
        probs.append('A and B loaded different staged runs')
    if probs:
        for p in probs:
            print('REFUSED: %s' % p)
        return {'status': 'REFUSED', 'problems': probs}, False
    out = {'failed': {}, 'reported': {}, 'missing': [], 'printed_worst': 0.0}
    for name in expected_files():
        p, q = os.path.join(adir, name), os.path.join(bdir, name)
        if not (os.path.exists(p) and os.path.exists(q)):
            out['missing'].append(name)
            continue
        if name.endswith('.json'):
            _rows(json.load(open(p)), json.load(open(q)), name, out)
            continue
        a, b = np.load(p, allow_pickle=True), np.load(q, allow_pickle=True)
        if set(a.files) != set(b.files):
            out['failed'][name + ':keys'] = 'A %s | B %s' % (sorted(a.files), sorted(b.files))
        for k in sorted(set(a.files) & set(b.files)):
            x, y, key = np.asarray(a[k]), np.asarray(b[k]), '%s:%s' % (name, k)
            if name.startswith('l4_') and k in L4_BITWISE:
                d = diff_arrays(x, y)
                if d:
                    out['failed'][key] = d
            elif name.startswith('l4_') and (k.startswith(('rms_', 'nrmse_')) or k == 'val_score'):
                ok, worst = printed(x, y)
                out['printed_worst'] = max(out['printed_worst'], worst)
                if not ok:
                    out['failed'][key] = '%.2f units of the 7th significant digit' % worst
            else:
                _reported(out, key, x, y)
    ok = not (out['failed'] or out['missing'])
    res = dict(status='PASS' if ok else 'FAIL' if out['failed'] else 'PARTIAL', **out)
    for k, v in list(out['failed'].items())[:20] + list(out['reported'].items())[:20]:
        print('    %s: %s' % (k, v))
    for m in out['missing']:
        print('MISSING: %s' % m)
    print('largest thesis-number difference: %.3f units of the 7th significant digit' % out['printed_worst'])
    return res, ok


if __name__ == '__main__':
    layer = sys.argv[1]
    if layer == 'l12':
        res, ok = run_l12(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif layer == 'l4':
        res, ok = run_l4(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit('unknown layer %s' % layer)
    with open(os.path.join(H.HERE, 'compare-d294-%s.json' % layer), 'w') as fh:
        json.dump(res, fh, indent=1, default=str)
    print('COMPARE_D294_%s %s' % (layer.upper(), 'PASS' if ok else res.get('status', 'FAIL')), flush=True)
    sys.exit(0 if ok else 1)
