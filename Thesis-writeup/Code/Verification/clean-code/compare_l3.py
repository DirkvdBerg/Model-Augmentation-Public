"""Layer 3, 3b and 3c comparison (design 6.3), used by compare.py.

Layer 3 compares two run folders: the state_dicts of both checkpoint pairs (hfn, encoder) and the
optimizer `state` entries where present; the `.npz` scalars and strings except `elapsed`; the
common `lbfgs_outcome` fields except `wall_s` (`reason` included); every val_hist array; config.json
on the shared keys except GIT, run_id, data_dir and NF_PROBE_PRINT. All bitwise, no fallback.
"""
import json
import os

import numpy as np
import torch

import hcommon as H
from compare import check_fingerprint, diff_arrays

L3_CASES = ['221-lowpass', '229-noisefree', '224-noisefree', '226-lowpass', '227-lowpass']
NPZ_SKIP = {'elapsed', 'orig_run_id', 'resumed_from'}
CFG_SKIP = {'GIT', 'run_id', 'data_dir', 'NF_PROBE_PRINT'}


def _cmp_tensors(prefix, a, b, out):
    """Recursive bitwise comparison of nested dict/list/tensor/scalar structures."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out[prefix + ':keys'] = 'golden %s | clean %s' % (sorted(map(str, a))[:20], sorted(map(str, b))[:20])
        for k in set(a) & set(b):
            _cmp_tensors('%s/%s' % (prefix, k), a[k], b[k], out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out[prefix + ':len'] = '%d vs %d' % (len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            _cmp_tensors('%s/%d' % (prefix, i), x, y, out)
    elif torch.is_tensor(a) and torch.is_tensor(b):
        d = diff_arrays(a.detach().cpu().numpy(), b.detach().cpu().numpy())
        if d:
            out[prefix] = d
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b)):
        out[prefix] = 'golden %r | clean %r' % (a, b)


def compare_pair(gbase, cbase, out, name):
    """A checkpoint pair <base>.pt/.npz on both sides."""
    for side, base in (('golden', gbase), ('clean', cbase)):
        if not (os.path.exists(base + '.pt') and os.path.exists(base + '.npz')):
            out[name + ':missing'] = '%s pair %s' % (side, base)
            return
    g, c = torch.load(gbase + '.pt', map_location='cpu'), torch.load(cbase + '.pt', map_location='cpu')
    if set(g) != set(c):
        out[name + '.pt:keys'] = 'golden %s | clean %s' % (sorted(g), sorted(c))
    for k in ('hfn', 'encoder'):
        _cmp_tensors('%s.pt/%s' % (name, k), dict(g[k]), dict(c[k]), out)
    if 'optimizer' in g and 'optimizer' in c:
        _cmp_tensors('%s.pt/optimizer' % name, g['optimizer'], c['optimizer'], out)
    gz, cz = np.load(gbase + '.npz', allow_pickle=True), np.load(cbase + '.npz', allow_pickle=True)
    for k in sorted(set(gz.files) | set(cz.files)):
        if k in NPZ_SKIP:
            continue
        if k not in gz.files or k not in cz.files:
            out['%s.npz:%s' % (name, k)] = 'only in %s' % ('golden' if k in gz.files else 'clean')
            continue
        if k == 'lbfgs_outcome':
            go, co = json.loads(str(gz[k])), json.loads(str(cz[k]))
            if (go is None) != (co is None):
                out['%s.npz:lbfgs_outcome' % name] = 'golden %r | clean %r' % (go, co)
                continue
            if go is None:
                continue
            for f in sorted(set(go) & set(co)):
                if f != 'wall_s' and go[f] != co[f]:
                    out['%s.npz:lbfgs_outcome.%s' % (name, f)] = 'golden %r | clean %r' % (go[f], co[f])
            continue
        d = diff_arrays(np.asarray(gz[k]), np.asarray(cz[k]))
        if d:
            out['%s.npz:%s' % (name, k)] = d + ' (golden %s | clean %s)' % (str(gz[k])[:120], str(cz[k])[:120])


def compare_runs(grun, crun, out, rid_g, rid_c):
    for suffix in ('', '_pre_lbfgs'):
        compare_pair(os.path.join(grun, 'gantry_ckpt_%s%s' % (rid_g, suffix)),
                     os.path.join(crun, 'gantry_ckpt_%s%s' % (rid_c, suffix)), out,
                     'final' if not suffix else 'pre_lbfgs')
    gv, cv = np.load(os.path.join(grun, 'val_hist.npz')), np.load(os.path.join(crun, 'val_hist.npz'))
    if set(gv.files) != set(cv.files):
        out['val_hist:keys'] = 'golden %s | clean %s' % (sorted(gv.files), sorted(cv.files))
    for k in sorted(set(gv.files) & set(cv.files)):
        d = diff_arrays(gv[k], cv[k])
        if d:
            out['val_hist:' + k] = d
    gc = json.load(open(os.path.join(grun, 'config.json')))
    cc = json.load(open(os.path.join(crun, 'config.json')))
    for k in sorted(set(gc) & set(cc)):
        if k not in CFG_SKIP and gc[k] != cc[k]:
            out['config.json:' + k] = 'golden %r | clean %r' % (gc[k], cc[k])
    return gc, cc


def run_l3(cases):
    res, ok_all = {}, True
    for case in cases or L3_CASES:
        gdir, cdir = os.path.join(H.GOLDEN, 'L3', case), os.path.join(H.CLEAN_OUT, 'L3', case)
        if not os.path.exists(os.path.join(cdir, 'meta.json')):
            res[case] = {'status': 'NO CLEAN OUTPUT'}
            ok_all = False
            print('%-14s L3 NO CLEAN OUTPUT' % case)
            continue
        gmeta, cmeta = json.load(open(os.path.join(gdir, 'meta.json'))), json.load(open(os.path.join(cdir, 'meta.json')))
        probs = check_fingerprint(gmeta, cmeta)
        if probs:
            res[case] = {'status': 'REFUSED (fingerprint)', 'problems': probs}
            ok_all = False
            print('%-14s L3 REFUSED %s' % (case, probs))
            continue
        out = {}
        rid = 'L3-' + case
        compare_runs(os.path.join(gdir, 'run'), os.path.join(cdir, 'run'), out, rid, rid)
        ok = not out
        ok_all &= ok
        res[case] = dict(status='PASS' if ok else 'FAIL', mismatch=out,
                         seconds=dict(golden=gmeta['seconds'], clean=cmeta['seconds']))
        print('%-14s L3 %s  (golden %.0f s, clean %.0f s)' % (case, res[case]['status'],
                                                              gmeta['seconds'], cmeta['seconds']), flush=True)
        for k, v in list(out.items())[:15]:
            print('    %s: %s' % (k, v))
    return res, ok_all


def run(layer, cases):
    if layer == 'l3':
        return run_l3(cases)
    if layer == 'l3b':
        import compare_l3bc
        return compare_l3bc.run_l3b(cases)
    if layer == 'l3c':
        import compare_l3bc
        return compare_l3bc.run_l3c(cases)
    raise SystemExit('unknown layer %s' % layer)
