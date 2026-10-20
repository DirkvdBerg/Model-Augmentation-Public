"""Compare the clean side against the mainline goldens (design 6.3), bitwise.

    python -u compare.py l12 [221-lowpass ...]      layers 1 and 2
    python -u compare.py l3  [221-lowpass ...]      layer 3
    python -u compare.py l3b [221-lowpass ...]      layer 3b
    python -u compare.py l3c [226-lowpass-i ...]    layer 3c

Before any array is read, the golden's fingerprint is recomputed (6.2): the 19 copied mainline
files, both manifests and every record the case read must hash as they did at capture, and the
clean side must have read the same records with the same libraries, threads and budget. Arrays are
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


def check_fingerprint(gmeta, cmeta):
    """Problems with the golden's fingerprint, [] when it is current and both sides match."""
    probs = []
    now = H.source_hashes()['mainline']
    for k, v in gmeta['sources']['mainline'].items():
        if now.get(k) != v:
            probs.append('mainline source changed since capture: %s' % k)
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
            r[layer] = dict(status='PASS' if ok else 'FAIL', **c,
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
