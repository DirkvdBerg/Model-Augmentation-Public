"""Layers 3b and 3c comparison (design 6.3), used by compare.py.

3b: outcome fields common to both sides except wall_s (reason included), the parameters after the
rejected polish (must equal the start state on each side, asserted in the run) and the full state
after it, bitwise. 3c: both pairs' state_dicts and optimizer entries, .npz except elapsed,
orig_run_id (the clean rerun keeps the folder's id) and resumed_from (compared by file name),
lbfgs_outcome as layer 3, val_hist arrays; plus the clean-only assertions written by the run and,
for ii and iv, the D-278 checks of both sides. The clean-only cases v, vi, vii (step 4c) have no
golden: their assertions are checked on a clean output made from the current clean tree.
"""
import json
import os

import numpy as np
import torch

import hcommon as H
from compare import check_fingerprint, d278_kept, output_problems
from compare_l3 import _cmp_tensors, compare_runs

L3B_CASES = ['221-lowpass', '226-lowpass']
L3C_CASES = ['226-lowpass-i', '226-lowpass-ii', '226-lowpass-iii', '226-lowpass-iv', '221-lowpass-i',
             '221-lowpass-ii']
L3C_CLEAN_ONLY = ['226-lowpass-v', '224-noisefree-vi', '226-lowpass-vii']


def _meta(d):
    return json.load(open(os.path.join(d, 'meta.json')))


def run_l3b(cases):
    res, ok_all = {}, True
    for case in cases or L3B_CASES:
        gdir, cdir = os.path.join(H.GOLDEN, 'L3b', case), os.path.join(H.CLEAN_OUT, 'L3b', case)
        if not os.path.exists(os.path.join(cdir, 'meta.json')):
            res[case] = {'status': 'NO CLEAN OUTPUT'}; ok_all = False
            continue
        probs = (check_fingerprint(_meta(gdir), _meta(cdir))
                 + output_problems(gdir, _meta(gdir), cdir, _meta(cdir)))
        if probs:
            res[case] = {'status': 'REFUSED (fingerprint or outputs)', 'problems': probs}; ok_all = False
            continue
        out = {}
        go, co = json.load(open(os.path.join(gdir, 'outcome.json'))), json.load(open(os.path.join(cdir, 'outcome.json')))
        for f in sorted(set(go['outcome']) & set(co['outcome'])):
            if f != 'wall_s' and go['outcome'][f] != co['outcome'][f]:
                out['outcome.' + f] = 'golden %r | clean %r' % (go['outcome'][f], co['outcome'][f])
        for side, o in (('golden', go), ('clean', co)):
            if not o['params_restored']:
                out[side + ':params_restored'] = False
            if o['outcome']['accepted']:
                out[side + ':accepted'] = 'the forced reject path was not taken'
        gs, cs = torch.load(os.path.join(gdir, 'state.pt')), torch.load(os.path.join(cdir, 'state.pt'))
        _cmp_tensors('state', gs, cs, out)
        ok = not out
        ok_all &= ok
        res[case] = dict(status='PASS' if ok else 'FAIL', mismatch=out, golden_pre_d278=d278_kept(_meta(gdir)),
                         seconds=dict(golden=_meta(gdir)['seconds'], clean=_meta(cdir)['seconds']))
        print('%-16s L3b %s' % (case, res[case]['status']), flush=True)
        for k, v in list(out.items())[:15]:
            print('    %s: %s' % (k, v))
    return res, ok_all


def run_l3c(cases):
    res, ok_all = {}, True
    for tag in cases or (L3C_CASES + L3C_CLEAN_ONLY):
        kind = tag.rsplit('-', 1)[1]
        gdir, cdir = os.path.join(H.GOLDEN, 'L3c', tag), os.path.join(H.CLEAN_OUT, 'L3c', tag)
        if not os.path.exists(os.path.join(cdir, 'meta.json')):
            res[tag] = {'status': 'NO CLEAN OUTPUT'}; ok_all = False
            continue
        out = {}
        if tag in L3C_CLEAN_ONLY:              # step 4c clean-only behaviour: no golden exists
            probs = (H.clean_tree_problems(_meta(cdir)['sources'].get('clean'))
                     + ['clean: ' + p for p in H.output_problems(cdir, _meta(cdir))])
            if probs:
                res[tag] = {'status': 'REFUSED (fingerprint or outputs)', 'problems': probs}; ok_all = False
                continue
            ca = json.load(open(os.path.join(cdir, 'clean_asserts.json')))
            if not ca['ok']:
                out['clean_asserts'] = ca
            ok = not out
            ok_all &= ok
            res[tag] = dict(status='PASS' if ok else 'FAIL', mismatch=out, clean_asserts=ca,
                            seconds=dict(clean=_meta(cdir)['seconds']))
            print('%-18s L3c %s (clean only)' % (tag, res[tag]['status']), flush=True)
            for k, v in list(out.items())[:15]:
                print('    %s: %s' % (k, v))
            continue
        # D-279: ii and iv run the D-278 paths, so their goldens must be the recaptured ones
        probs = (check_fingerprint(_meta(gdir), _meta(cdir), allow_d278=kind not in ('ii', 'iv'))
                 + output_problems(gdir, _meta(gdir), cdir, _meta(cdir)))
        if probs:
            res[tag] = {'status': 'REFUSED (fingerprint or outputs)', 'problems': probs}; ok_all = False
            continue
        case = tag.rsplit('-', 1)[0]
        compare_runs(os.path.join(gdir, 'run'), os.path.join(cdir, 'run'), out,
                     'L3c-' + tag, 'L3-' + case)
        out = {k: v for k, v in out.items() if not k.startswith('config.json:')}   # 3c: not compared
        for suffix in ('', '_pre_lbfgs'):
            gp = os.path.join(gdir, 'run', 'gantry_ckpt_L3c-%s%s.npz' % (tag, suffix))
            cp = os.path.join(cdir, 'run', 'gantry_ckpt_L3-%s%s.npz' % (case, suffix))
            if os.path.exists(gp) and os.path.exists(cp):
                gz, cz = np.load(gp, allow_pickle=True), np.load(cp, allow_pickle=True)
                if 'resumed_from' in gz.files and 'resumed_from' in cz.files:
                    gn = os.path.basename(str(gz['resumed_from']))
                    cn = os.path.basename(str(cz['resumed_from']))
                    if gn.replace('L3c-' + tag, 'X').replace('L3-' + case, 'X') != cn.replace('L3-' + case, 'X'):
                        out['resumed_from%s' % suffix] = 'golden %s | clean %s' % (gn, cn)
        ca = json.load(open(os.path.join(cdir, 'clean_asserts.json')))
        if not ca['ok']:
            out['clean_asserts'] = ca
        checks = {}
        if kind in ('ii', 'iv'):               # D-278 on both sides, written by run_l3bc
            for side, d in (('golden', gdir), ('clean', cdir)):
                checks[side] = json.load(open(os.path.join(d, 'd278_checks.json')))
                if not checks[side]['ok']:
                    out['d278_checks:' + side] = checks[side]
        ok = not out
        ok_all &= ok
        res[tag] = dict(status='PASS' if ok else 'FAIL', mismatch=out, golden_pre_d278=d278_kept(_meta(gdir)),
                        d278_checks=checks,
                        seconds=dict(golden=_meta(gdir)['seconds'], clean=_meta(cdir)['seconds']))
        print('%-18s L3c %s' % (tag, res[tag]['status']), flush=True)
        for k, v in list(out.items())[:15]:
            print('    %s: %s' % (k, v))
    return res, ok_all
