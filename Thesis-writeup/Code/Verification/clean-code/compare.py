"""Compare the clean side against the mainline goldens (design 6.3), bitwise.

    python -u compare.py l12 [221-lowpass ...]      layers 1 and 2
    python -u compare.py l3  [221-lowpass ...]      layer 3
    python -u compare.py l3b [221-lowpass ...]      layer 3b
    python -u compare.py l3c [226-lowpass-i ...]    layer 3c

Before any array is read, the golden's fingerprint is recomputed (6.2): the 19 copied mainline
files, both manifests and every record the case read must hash as they did at capture, and the
clean side must have read the same records with the same libraries, threads and budget, from the
CURRENT clean tree (H2: every file under Clean/ hashes as recorded by the clean output). Arrays are
compared on dtype, shape and bytes; a difference is reported with its max abs and relative error
and is a FAIL (the 1e-12 fallback of layers 1-2 is applied only by the judge, never here).
Writes compare-<layer>.json next to this script.
"""
import json
import os
import sys

import numpy as np

import hcommon as H

L12_CASES = ['%d-%s' % (r, n) for n in ('lowpass', 'noisefree')
             for r in (221, 222, 223, 224, 226, 227, 229, 230)]
L3_CASES = ['221-lowpass', '229-noisefree', '224-noisefree', '226-lowpass', '227-lowpass']


# D-279: training.py before and after the D-278 commit (b4a985b), as captured (CRLF working tree).
# The two differ only by D-278, which reaches a polish-only start and a resume with the Adam budget
# spent; the 3c cases that run those paths (ii, iv) were recaptured. Every other golden may carry
# exactly this old hash while the file has exactly this new one.
D278_FILE = 'scripts/gantry/gantry_dynamic/training.py'
D278_PAIR = ('43bbf868595d4f67985fd8141700baf9a6b84c5b7b6cf74ed1eb6ebce627325a',
             '9ee2a53ea6014c0c00526333eeeae71d016b047dd91dc2586db875ad9edacf3f')


#: D-281 (O8): golden fields the optimised clean code no longer has, with the item that dropped them.
#: Skipped only when absent on the clean side, and listed as 'dropped'. Their content stays covered:
#: bank.ystd and bank.stdu enter M_state and M_error (B41); the validation rows follow the val_files
#: order the names came from (B51).
DROPPED_IN_CLEAN = {'bank/ystd': 'B41', 'bank/stdu': 'B41', 'info:sim/val_labels': 'B51'}


def d278_kept(gmeta):
    """True when the golden was captured before D-278 (accepted only through D278_PAIR)."""
    return gmeta['sources']['mainline'].get(D278_FILE) == D278_PAIR[0]


def check_fingerprint(gmeta, cmeta, allow_d278=True):
    """Problems with the golden's fingerprint, [] when it is current, both sides match and the
    clean output was made from the current clean tree (H2)."""
    probs = []
    now = H.source_hashes()['mainline']
    for k, v in gmeta['sources']['mainline'].items():
        if now.get(k) != v and not (allow_d278 and k == D278_FILE and (v, now.get(k)) == D278_PAIR):
            probs.append('mainline source changed since capture: %s' % k)
    probs += H.clean_tree_problems(cmeta['sources'].get('clean'))
    files = gmeta['files']['train'] + gmeta['files']['val'] + gmeta['files']['test']
    data_now = H.data_hashes(gmeta['data_dir'], files)
    for k, v in gmeta['data'].items():
        if data_now.get(k) != v:
            probs.append('data changed since capture: %s' % k)
    for key in ('data', 'files', 'libraries', 'budget'):
        if gmeta[key] != cmeta[key]:
            probs.append('%s differs between the sides' % key)
    for side, m in (('golden', gmeta), ('clean', cmeta)):
        t = m['threads']
        if not (t['torch_num_threads'] == 1 and t['OMP_NUM_THREADS'] == '1' and t['MKL_NUM_THREADS'] == '1'):
            probs.append('%s side not single-threaded: %s' % (side, t))
    if os.path.normcase(gmeta['data_dir']) != os.path.normcase(cmeta['data_dir']):
        probs.append('data_dir differs: %s vs %s' % (gmeta['data_dir'], cmeta['data_dir']))
    return probs


def diff_arrays(a, b):
    """None when bitwise equal, else a description."""
    if a.dtype != b.dtype:
        return 'dtype %s vs %s' % (a.dtype, b.dtype)
    if a.shape != b.shape:
        return 'shape %s vs %s' % (a.shape, b.shape)
    if a.tobytes() == b.tobytes():
        return None
    if a.dtype.kind in 'fc':
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        ref = float(np.nanmax(np.abs(a))) if a.size else 0.0
        mx = float(np.nanmax(d)) if d.size else 0.0
        return 'values: max abs %.3e, rel %.3e, %d of %d entries differ' % (
            mx, mx / ref if ref else float('inf'), int((a != b).sum()), a.size)
    return 'values: %d of %d entries differ' % (int((a != b).sum()), a.size)


def compare_npz(gdir, cdir, stem):
    out = {'mismatch': {}, 'missing_clean': [], 'extra_clean': [], 'n_compared': 0}
    g, c = np.load(os.path.join(gdir, stem + '.npz')), np.load(os.path.join(cdir, stem + '.npz'))
    gj = json.load(open(os.path.join(gdir, stem + '.json')))
    cj = json.load(open(os.path.join(cdir, stem + '.json')))
    gk, ck = set(g.files), set(c.files)
    out['dropped'] = sorted(((gk - ck) | {'info:' + k for k in set(gj['info']) - set(cj['info'])})
                            & set(DROPPED_IN_CLEAN))
    gk -= set(out['dropped'])
    out['missing_clean'] = sorted(gk - ck)
    out['extra_clean'] = sorted(ck - gk)
    for k in sorted(gk & ck):
        d = diff_arrays(g[k], c[k])
        out['n_compared'] += 1
        if d:
            out['mismatch'][k] = d
    for k in sorted(set(gj['hashed']) | set(cj['hashed'])):
        out['n_compared'] += 1
        if gj['hashed'].get(k, {}).get('sha256') != cj['hashed'].get(k, {}).get('sha256') \
                or gj['hashed'].get(k, {}).get('shape') != cj['hashed'].get(k, {}).get('shape'):
            out['mismatch'][k] = 'hash differs (golden %s, clean %s)' % (gj['hashed'].get(k), cj['hashed'].get(k))
    for k in sorted(set(gj['info']) | set(cj['info'])):
        if 'info:' + k in out['dropped']:
            continue
        out['n_compared'] += 1
        if gj['info'].get(k) != cj['info'].get(k):
            out['mismatch']['info:' + k] = 'golden %s | clean %s' % (str(gj['info'].get(k))[:200],
                                                                    str(cj['info'].get(k))[:200])
    return out


def run_l12(cases):
    res, ok_all = {}, True
    for case in cases:
        r = {}
        for layer, stem in (('L1', 'L1'), ('L2', 'L2')):
            gdir = os.path.join(H.GOLDEN, layer, case)
            cdir = os.path.join(H.CLEAN_OUT, layer, case)
            if not os.path.exists(os.path.join(cdir, 'meta.json')):
                r[layer] = {'status': 'NO CLEAN OUTPUT'}
                ok_all = False
                continue
            gmeta = json.load(open(os.path.join(gdir, 'meta.json')))
            cmeta = json.load(open(os.path.join(cdir, 'meta.json')))
            probs = check_fingerprint(gmeta, cmeta)
            if probs:
                r[layer] = {'status': 'REFUSED (fingerprint)', 'problems': probs}
                ok_all = False
                continue
            c = compare_npz(gdir, cdir, stem)
            ok = not (c['mismatch'] or c['missing_clean'] or c['extra_clean'])
            ok_all &= ok
            r[layer] = dict(status='PASS' if ok else 'FAIL', **c, golden_pre_d278=d278_kept(gmeta),
                            seconds=dict(golden=gmeta['timings']['total'], clean=cmeta['timings']['total']))
        res[case] = r
        print('%-14s L1 %-6s L2 %-6s' % (case, r['L1']['status'], r['L2']['status']), flush=True)
        for layer in ('L1', 'L2'):
            for k, v in list(r[layer].get('mismatch', {}).items())[:12]:
                print('    %s %s: %s' % (layer, k, v))
    return res, ok_all


if __name__ == '__main__':
    layer = sys.argv[1]
    cases = sys.argv[2:]
    if layer == 'l12':
        res, ok = run_l12(cases or L12_CASES)
    else:
        import compare_l3
        res, ok = compare_l3.run(layer, cases)
    with open(os.path.join(H.HERE, 'compare-%s.json' % layer), 'w') as fh:
        json.dump(res, fh, indent=1, default=str)
    print('COMPARE_%s %s' % (layer.upper(), 'PASS' if ok else 'FAIL'), flush=True)
    sys.exit(0 if ok else 1)
